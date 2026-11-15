import os,subprocess,time,json,urllib.request,pathlib,signal
r=pathlib.Path('/data/prefixb-105-gpu1-20261114')
env=os.environ.copy()
env.update(PATH='/opt/conda/bin:/opt/mxdriver/bin:/opt/maca/bin:'+env.get('PATH',''),MACA_PATH='/opt/maca',LD_LIBRARY_PATH='/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib',VLLM_PLUGINS='fl',PYTHONNOUSERSITE='1',TORCHINDUCTOR_COMPILE_THREADS='1',CUDA_VISIBLE_DEVICES='1')
def status(**kw): (r/'status.json').write_text(json.dumps(dict(timestamp=time.time(),**kw)))
args=[str(r/'b-env/bin/python'),'-u','-B','-m','vllm.entrypoints.cli.main','serve','/workspace/MiniCPM5-2B','--port','9031','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
s=subprocess.Popen(args,env=env,stdout=open(r/'server.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
status(state='starting',pid=s.pid,argv=args)
healthy=False
for i in range(180):
 if s.poll() is not None:break
 try:
  if urllib.request.urlopen('http://127.0.0.1:9031/health',timeout=2).status==200:healthy=True;break
 except Exception:pass
 time.sleep(2)
if not healthy:
 status(state='server_not_ready',server_rc=s.poll())
 if s.poll() is None:os.killpg(s.pid,signal.SIGTERM)
 raise SystemExit(1)
env.update(VLLM_PLUGINS='',HF_DATASETS_OFFLINE='1',HF_HUB_OFFLINE='1',MODELSCOPE_OFFLINE='1')
args=['/opt/conda/bin/evalscope','eval','--model','minicpm','--api-url','http://127.0.0.1:9031/v1/chat/completions','--api-key','EMPTY','--eval-type','openai_api','--datasets','math_500','--dataset-args',json.dumps({'math_500':{'dataset_id':'/workspace/evalscope-datasets/math_500','subset_list':['Level 3']}}),'--eval-batch-size','8','--timeout','3600','--generation-config',json.dumps({'temperature':1.0,'top_p':0.95,'max_tokens':32768}),'--work-dir',str(r/'eval'),'--ignore-errors']
c=subprocess.Popen(args,env=env,stdout=open(r/'client.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
status(state='evaluating',server_pid=s.pid,client_pid=c.pid,argv=args)
try:
 rc=c.wait(timeout=3650);status(state='client_finished_needs_accuracy_audit',client_rc=rc,server_rc=s.poll())
except subprocess.TimeoutExpired:
 os.killpg(c.pid,signal.SIGTERM);status(state='client_timeout')
finally:
 if s.poll() is None:os.killpg(s.pid,signal.SIGTERM)
