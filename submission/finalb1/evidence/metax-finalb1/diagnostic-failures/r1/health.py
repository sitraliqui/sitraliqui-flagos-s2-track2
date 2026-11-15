import datetime, json

def mark(step): print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),step=step)),flush=True)
mark('torch import begin')
import torch
mark('torch import complete')
mark('device enumerate begin'); n=torch.cuda.device_count(); assert n==1, 'Expected exactly one GPU'; mark('device enumerate complete: '+str(n))
mark('properties begin'); props=torch.cuda.get_device_properties(0); mark('properties complete'); assert 'C500' in props.name, props.name
mark('device='+props.name+' total_memory='+str(props.total_memory)+' torch='+torch.__version__+' cuda='+str(torch.version.cuda))
mark('init begin'); torch.cuda.init(); mark('init complete')
mark('empty begin'); x=torch.empty((16,16),device='cuda'); mark('empty complete')
mark('fill begin'); x.fill_(1); mark('fill complete')
mark('sync begin'); torch.cuda.synchronize(); mark('sync complete')
for name,dtype in [('FP32',torch.float32),('BF16',torch.bfloat16)]:
    mark(name+' matrix begin'); a=torch.ones((16,16),device='cuda',dtype=dtype); c=a@a; mark(name+' matrix submitted')
    mark(name+' sync begin'); torch.cuda.synchronize(); mark(name+' sync complete')
    assert torch.equal(c.cpu().float(),torch.full((16,16),16.0)); mark(name+' numeric passed')
mark('health passed')


