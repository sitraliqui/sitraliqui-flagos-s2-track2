"""Submission convergence only: merged installation, diagnostic hit, official accuracy/perf."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='1', VLLM_PLUGINS='fl', PYTHONNOUSERSITE='1',
           TORCHINDUCTOR_COMPILE_THREADS='1', MACA_PATH='/opt/maca',
           PATH='/opt/conda/bin:/opt/mxdriver/bin:/opt/maca/bin:'+os.environ.get('PATH',''),
           LD_LIBRARY_PATH='/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib')
SERVE = ['-u','-B','-m','vllm.entrypoints.cli.main','serve','/workspace/MiniCPM5-2B',
         '--port','9031','--served-model-name','minicpm','--gpu-memory-utilization','0.85',
         '--max-model-len','131072']
STATE = {'started_unix':time.time(), 'stage':'preflight'}
DEADLINE = json.loads((ROOT/'pin.json').read_text())['hard_stop_unix']


def save(**changes):
    STATE.update(changes, updated_unix=time.time())
    temporary = ROOT/'status.tmp'
    temporary.write_text(json.dumps(STATE,indent=2))
    temporary.replace(ROOT/'status.json')


def run(label, command, seconds, env=ENV, cwd=ROOT):
    seconds = min(seconds, DEADLINE-time.time()-90)
    if seconds <= 10:
        raise TimeoutError('Archive and cleanup reserve reached')
    with (ROOT/(label+'.log')).open('xb') as log:
        subprocess.run(command,env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,
                       stderr=subprocess.STDOUT,timeout=seconds,check=True)


def stop(server):
    if server and server.poll() is None:
        os.killpg(server.pid,signal.SIGTERM)
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid,signal.SIGKILL)
            server.wait(timeout=10)


def start_server(python, label, env):
    for n in ('tcp','tcp6'):
        for line in Path('/proc/net',n).read_text().splitlines()[1:]:
            f=line.split()
            if int(f[1].rsplit(':',1)[1],16)==9031 and f[3]=='0A':
                raise RuntimeError('Pre-existing service on port9031')
    log = (ROOT/(label+'.log')).open('xb')
    server = subprocess.Popen([str(python),*SERVE],env=env,stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    save(server_pid=server.pid, server_argv=[str(python),*SERVE])
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        end=min(time.time()+360,DEADLINE-120)
        while time.time()<end:
            if server.poll() is not None:
                raise RuntimeError('Server stopped during initialization')
            try:
                with opener.open('http://127.0.0.1:9031/health',timeout=2) as response:
                    if response.status==200:
                        return server
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError('Server readiness deadline')
    except BaseException:
        stop(server)
        raise


def main():
    server=None
    save()
    try:
        run('health',['/opt/conda/bin/python','-u','-B',str(ROOT/'health.py')],120)
        run('venv',['/opt/conda/bin/python','-m','venv','--system-site-packages',str(ROOT/'merged-env')],60)
        python=ROOT/'merged-env/bin/python'
        wheels=sorted((ROOT/'wheels').glob('*.whl'))
        if len(wheels)!=2:
            raise ValueError('Exactly two merged wheels required')
        run('install',[str(python),'-m','pip','install','--no-index','--no-deps','--no-compile',*[str(w) for w in wheels]],120)
        run('installation-audit',[str(python),'-B',str(ROOT/'verify_install.py')],30)
        save(stage='matched_baseline_install_verified')
        run('installation-audit-formal',[str(python),'-B',str(ROOT/'verify_install.py')],30)
        save(stage='formal_startup',diagnostic_observer_removed=True,variant='B-independent-4k-verification')
        server=start_server(python,'formal-server',ENV)
        client_env=dict(ENV,VLLM_PLUGINS='',HF_DATASETS_OFFLINE='1',HF_HUB_OFFLINE='1',MODELSCOPE_OFFLINE='1')
        previous=Path('/data/competition-final-prefixb-20261115-v1-r3')
        reference=json.loads((previous/'accuracy-audit.json').read_text())
        assert reference['passed'] and reference['count']==105 and reference['correct']==102
        expected={'7271142a9a1d0fec441a2b7e3d24cc0e44c8f8b2b21b41e865fa3cffbe6da0f6','307a615dba8a80edcbafa51cf168059b89d25a130b8d9caa3a795d281ed6f8be'}
        assert {hashlib.sha256(w.read_bytes()).hexdigest() for w in wheels}==expected
        save(stage='performance',accuracy_reference={'source':str(previous/'accuracy-audit.json'),'result':reference,'new_accuracy_run':False})
        script=ROOT/'official-benchmark.py'
        if hashlib.sha256(script.read_bytes()).hexdigest()!='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565':
            raise ValueError('Official benchmark script differs')
        perf=ROOT/'performance'
        perf.mkdir()
        argv=['/opt/conda/bin/python','-u',str(script),'--model','/workspace/MiniCPM5-2B',
              '--served-model-name','minicpm','--port','9031','--test-cases','[[4096,1024,64,256]]']
        save(performance_argv=argv)
        run('performance-client',argv,3600,env=client_env,cwd=perf)
        sys.path.insert(0,str(ROOT))
        from csv_validator import validate_benchmark
        raw,=perf.glob('benchmark_results/raw_runs_*.csv')
        summary,=perf.glob('benchmark_results/summary_*.csv')
        result=validate_benchmark(raw,summary,script,('4k',))
        (ROOT/'performance-audit.json').write_text(json.dumps(result,indent=2))
        save(stage='completed',performance=result)
    except BaseException as error:
        save(stage='failed',error=repr(error))
        raise
    finally:
        stop(server)
        save(server_stopped=server is None or server.poll() is not None)


if __name__=='__main__':
    main()
