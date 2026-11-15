"""Exact wheel member and installed distribution check without initializing a GPU."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

root=Path(__file__).resolve().parent
result=[]
for wheel in sorted((root/'wheels').glob('*.whl')):
    package='vllm_fl' if wheel.name.startswith('vllm_plugin_fl') else 'flag_gems'
    dist=importlib.metadata.distribution('vllm-plugin-fl' if package=='vllm_fl' else 'flag-gems')
    files=0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith(package+'/') and not name.endswith('/'):
                if Path(dist.locate_file(name)).read_bytes()!=archive.read(name):
                    raise ValueError('Installed file mismatch '+name)
                files+=1
    result.append({'package':package,'version':dist.version,'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'files':files})
if 'torch' in sys.modules:
    raise RuntimeError('Static installation check unexpectedly initialized torch')
print(json.dumps(result))
