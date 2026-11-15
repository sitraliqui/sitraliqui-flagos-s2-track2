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
        # Diagnostic-only source witness; byte-for-byte restored before official tests.
        site=ROOT/'merged-env/lib/python3.12/site-packages'
        target=site/'vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py'
        original=target.read_bytes()
        assert hashlib.sha256(original).hexdigest()=='f96b3a72c52c281ef1983acda6dd14120857334a600dec4366f607aa7b5acf20'
        marker='                    ).cumsum(0, dtype=torch.int32)\n'
        witness="                    __import__('pathlib').Path("+repr(str(ROOT/'hit-source.json'))+").write_text(__import__('json').dumps(dict(pid=__import__('os').getpid(), num_prefills=int(attn_metadata.num_prefills), num_decode_tokens=int(attn_metadata.num_decode_tokens), implementation=type(self).__name__, source=__file__, diagnostic_only=True)))\n"
        text=original.decode()
        assert text.count(marker)==1
        target.write_text(text.replace(marker,marker+witness))
        save(stage='diagnostic_hit',diagnostic_method='temporary-source-witness-restored-before-formal', diagnostic_source_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        try:
            server=start_server(python,'diagnostic-server',ENV)
            body=json.dumps({'model':'minicpm','messages':[{'role':'user','content':'Calculate 17 + 25. Explain briefly.'}],'temperature':0,'max_tokens':32}).encode()
            req=urllib.request.Request('http://127.0.0.1:9031/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=120) as response:
                answer=json.load(response)
            (ROOT/'diagnostic-response.json').write_text(json.dumps(answer))
            if not (ROOT/'hit-source.json').exists() or not answer.get('choices'):
                raise RuntimeError('Merged Prefix B source branch hit not observed')
            save(hit_files=['hit-source.json'])
        finally:
            stop(server)
            server=None
            target.write_bytes(original)
        run('installation-audit-formal',[str(python),'-B',str(ROOT/'verify_install.py')],30)
        save(stage='formal_startup',diagnostic_observer_removed=True)
        server=start_server(python,'formal-server',ENV)
        client_env=dict(ENV,VLLM_PLUGINS='',HF_DATASETS_OFFLINE='1',HF_HUB_OFFLINE='1',MODELSCOPE_OFFLINE='1')
        eval_argv=['/opt/conda/bin/evalscope','eval','--model','minicpm','--api-url','http://127.0.0.1:9031/v1/chat/completions',
                   '--api-key','EMPTY','--eval-type','openai_api','--datasets','math_500','--dataset-args',
                   json.dumps({'math_500':{'dataset_id':'/workspace/evalscope-datasets/math_500','subset_list':['Level 3']}}),
                   '--eval-batch-size','8','--timeout','3600','--generation-config',
                   json.dumps({'temperature':1.0,'top_p':0.95,'max_tokens':32768}),
                   '--work-dir',str(ROOT/'accuracy'),'--ignore-errors']
        save(stage='accuracy',accuracy_argv=eval_argv)
        run('accuracy-client',eval_argv,3650,env=client_env)
        sys.path.insert(0,str(ROOT))
        from accuracy_gate import check
        accuracy=check(ROOT/'accuracy')
        (ROOT/'accuracy-audit.json').write_text(json.dumps(accuracy,indent=2))
        save(stage='performance',accuracy=accuracy)
        script=ROOT/'official-benchmark.py'
        if hashlib.sha256(script.read_bytes()).hexdigest()!='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565':
            raise ValueError('Official benchmark script differs')
        perf=ROOT/'performance'
        perf.mkdir()
        argv=['/opt/conda/bin/python','-u',str(script),'--model','/workspace/MiniCPM5-2B',
              '--served-model-name','minicpm','--port','9031','--test-cases','[[4096,1024,64,256],[16384,1024,64,128]]']
        save(performance_argv=argv)
        run('performance-client',argv,3600,env=client_env,cwd=perf)
        from csv_validator import validate_benchmark
        raw,=perf.glob('benchmark_results/raw_runs_*.csv')
        summary,=perf.glob('benchmark_results/summary_*.csv')
        result=validate_benchmark(raw,summary,script,('4k','16k'))
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
