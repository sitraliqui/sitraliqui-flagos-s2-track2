"""Diagnostic only: observe actual cat invocation across engine threads."""
import functools,hashlib,json,os,sys,time
from pathlib import Path
OUTPUT=os.environ.get('FINAL_HIT_DIR')
if OUTPUT:
 import torch
 original_cat=torch.cat
 seen=set()
 @functools.wraps(original_cat)
 def observed_cat(*args,**kwargs):
  result=original_cat(*args,**kwargs)
  caller=sys._getframe(1)
  name=caller.f_code.co_filename.replace('\\','/')
  if os.getpid() not in seen and caller.f_code.co_name=='forward' and name.endswith('/vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py'):
   meta=caller.f_locals.get('attn_metadata')
   if meta is not None and meta.num_prefills>0:
    seen.add(os.getpid())
    row=dict(pid=os.getpid(),unix=time.time(),source=name,line=caller.f_lineno,source_sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest(),num_prefills=int(meta.num_prefills),num_decode_tokens=int(meta.num_decode_tokens),implementation=type(caller.f_locals['self']).__name__,output_shape=list(result.shape),dtype=str(result.dtype),diagnostic_only=True)
    (Path(OUTPUT)/('hit-'+str(os.getpid())+'.json')).write_text(json.dumps(row))
  return result
 torch.cat=observed_cat
