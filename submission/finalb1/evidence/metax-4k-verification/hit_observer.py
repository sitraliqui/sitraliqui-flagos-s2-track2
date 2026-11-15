"""Trace one real vector-prefix source line in the separate diagnostic startup."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

OUTPUT=os.environ.get('FINAL_HIT_DIR')
if OUTPUT:
    seen=set()
    def observe(frame,event,arg):
        name=frame.f_code.co_filename.replace('\\','/')
        target=name.endswith('/vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py')
        if event=='call':
            return observe if target and frame.f_code.co_name=='forward' else None
        if target and event=='line' and frame.f_lineno==801 and os.getpid() not in seen:
            # Line 801 is inside the new torch.cat prefix construction.
            seen.add(os.getpid())
            meta=frame.f_locals.get('attn_metadata')
            if meta is not None and meta.num_prefills>0:
                row={'pid':os.getpid(),'unix':time.time(),'source':name,'line':801,
                     'source_sha256':hashlib.sha256(Path(name).read_bytes()).hexdigest(),
                     'num_prefills':int(meta.num_prefills),'num_decode_tokens':int(meta.num_decode_tokens),
                     'implementation':type(frame.f_locals['self']).__name__,'diagnostic_only':True}
                (Path(OUTPUT)/('hit-'+str(os.getpid())+'.json')).write_text(json.dumps(row))
        return observe if target else None
    sys.settrace(observe)
