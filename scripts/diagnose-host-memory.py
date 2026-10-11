"""Read-only staged import/context diagnostics. Does not load model weights."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def snapshot():
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for source in ['smaps_rollup', 'status']:
        try:
            lines = (Path('/proc/self')/source).read_text().splitlines()
            fields = {}
            for line in lines:
                key, _, value = line.partition(':')
                if key in ['Rss', 'Pss', 'Pss_Anon', 'Pss_File', 'Pss_Shmem', 'Private_Clean', 'Private_Dirty', 'Shared_Clean', 'Shared_Dirty', 'VmRSS', 'VmHWM', 'VmSize']:
                    fields[key] = value.strip()
            result[source] = fields
        except OSError:
            result[source] = None
    for prefix in [Path('/sys/fs/cgroup/memory'), Path('/sys/fs/cgroup')]:
        values = {}
        for filename in ['memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.max_usage_in_bytes', 'memory.current', 'memory.max', 'memory.stat', 'memory.oom_control', 'memory.events']:
            try:
                values[filename] = (prefix/filename).read_text().strip()
            except OSError:
                pass
        if values:
            result['cgroup'] = values
            break
    return result


def worker(case):
    def mark(name):
        print(json.dumps({'stage': name, **snapshot()}), flush=True)
    mark('python_start')
    if case == 'empty':
        return
    import torch
    mark('import_torch_done')
    if case == 'torch_init':
        torch.cuda.init()
        mark('cuda_init_done')
        x = torch.empty(1, device='cuda')
        torch.cuda.synchronize()
        mark('one_element_gpu_tensor_done')
        del x
    elif case in ['vllm_platform', 'plugin_register']:
        import vllm
        mark('import_vllm_done')
        if case == 'vllm_platform':
            from vllm.platforms import current_platform
            mark('resolve_current_platform_done')
            current_platform.get_device_capability()
            mark('device_capability_done')
        else:
            import vllm_fl
            mark('import_plugin_done')
            vllm_fl.register()
            mark('plugin_register_done')
    mark('worker_end')


def process_memory(pid):
    try:
        data = (Path('/proc')/str(pid)/'smaps_rollup').read_text().splitlines()
        return {s.split(':')[0]: int(s.split(':')[1].split()[0]) for s in data if s.split(':')[0] in ['Rss', 'Pss', 'Pss_Anon', 'Pss_File', 'Pss_Shmem']}
    except (OSError, ValueError):
        return {}


def run_suite(out, client_only=False):
    out.mkdir(parents=True, exist_ok=True)
    import importlib.metadata
    packages = {}
    for name in ['torch','vllm','vllm-plugin-fl','flag-gems','triton']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    (out/'environment.json').write_text(json.dumps({'python':sys.version,'packages':packages,'runtime_environment':{k:os.environ.get(k) for k in ['MACA_PATH','LD_LIBRARY_PATH','VLLM_PLUGINS','CUDA_VISIBLE_DEVICES','MACA_VISIBLE_DEVICES']}},indent=2)+'\n')
    # Independent processes prevent allocations from one phase contaminating another.
    cases = [('empty', False), ('torch_import', True), ('torch_import', False), ('torch_init', False), ('vllm_platform', False), ('plugin_register', False), ('cli_help', False), ('cli_help', True)]
    if client_only:
        cases = [('cli_help', True)]
    records = []
    for case, hidden in cases:
        label = case + ('_gpu_hidden' if hidden else '_gpu_visible')
        env = os.environ.copy()
        if hidden:
            env.update(CUDA_VISIBLE_DEVICES='-1', MACA_VISIBLE_DEVICES='-1')
        if client_only:
            env['VLLM_PLUGINS']=''
        command = ['vllm', 'bench', 'serve', '--help'] if case == 'cli_help' else [sys.executable, str(Path(__file__).resolve()), '--worker', case]
        before = snapshot()
        started = time.monotonic()
        samples = []
        reason = None
        with (out/(label+'.log')).open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            while child.poll() is None:
                sample = {'elapsed_s': time.monotonic()-started, 'process_kib': process_memory(child.pid), **snapshot()}
                samples.append(sample)
                cg = sample.get('cgroup', {})
                usage = cg.get('memory.current', cg.get('memory.usage_in_bytes'))
                if time.monotonic()-started > 180:
                    reason = 'case_timeout_180_seconds'
                elif usage and usage.isdigit() and int(usage) > 54 * 1024**3:
                    reason = 'container_memory_guard_54_GiB'
                if reason:
                    import signal
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                time.sleep(0.25)
        record = {'case': case, 'gpu_hidden': hidden, 'client_plugins_empty':client_only, 'exit_code': child.returncode, 'stop_reason': reason, 'elapsed_s': time.monotonic()-started, 'before': before, 'after': snapshot(), 'peak_process_kib': {key: max((s['process_kib'].get(key, 0) for s in samples), default=0) for key in ['Rss', 'Pss', 'Pss_Anon', 'Pss_File', 'Pss_Shmem']}}
        with (out/(label+'-samples.jsonl')).open('w') as f:
            for row in samples:
                f.write(json.dumps(row)+'\n')
        records.append(record)
        (out/'summary.json').write_text(json.dumps({'purpose': 'staged_host_memory_diagnostic_not_official_benchmark', 'cases': records}, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    (out/'finished.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker')
    parser.add_argument('--output', default='/workspace/memory-diagnostic-20261011')
    parser.add_argument('--client-only',action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        run_suite(Path(args.output),args.client_only)
