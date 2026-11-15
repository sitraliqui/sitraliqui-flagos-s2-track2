"""Linux detached-process supervisor. No SSH, platform operations or credentials."""
import argparse, json, math, os, signal, socket, subprocess, sys, time
from pathlib import Path
from owned_processes import proc_rows, track, subreaper

def validate(pin, hostname=None, boot=None, now=None):
    for key in ('platform','role','instance_id','owner','run_id','lease_id','hostname','boot_id','hard_stop_unix'):
        if not pin.get(key): raise ValueError('Missing pin: '+key)
    if pin['platform']!='metax' or pin['role']!='competition': raise ValueError('Wrong platform/role')
    if pin['instance_id'] not in ('LAJNBONXYAKBAYPK',): raise ValueError('Unapproved instance')
    if pin['hostname']!=(hostname or socket.gethostname()): raise ValueError('Hostname mismatch')
    if pin['boot_id']!=(boot or Path('/proc/sys/kernel/random/boot_id').read_text().strip()): raise ValueError('Boot mismatch')
    if type(pin['hard_stop_unix']) not in (int,float) or not math.isfinite(pin['hard_stop_unix']): raise ValueError('Invalid hard stop')
    remaining=pin['hard_stop_unix']-(time.time() if now is None else now)
    if remaining<20: raise ValueError('Insufficient cleanup reserve')
    return remaining

def main():
    p=argparse.ArgumentParser(); p.add_argument('--pin',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--memory-guard',action='store_true'); p.add_argument('--seconds',type=float,default=120); p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
    pin=json.loads(a.pin.read_text()); remaining=validate(pin)
    if not a.command or not math.isfinite(a.seconds) or a.seconds<=0: raise ValueError('Command and positive deadline required')
    a.out.mkdir(parents=True,exist_ok=False)
    receipt=dict(pin=pin,started_unix=time.time(),command=a.command,cleanup_complete=False)
    subreaper(); known={}; child=None; leader=None; start=None; guard=None
    stop=time.monotonic()+min(a.seconds,remaining-15)
    interrupted=[]
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(s,lambda sig,frame: interrupted.append(sig))
    try:
        if a.memory_guard:
            from memory_guard import MemoryGuard
            guard=MemoryGuard(a.out/'memory'); guard.start(); guard.check()
        with (a.out/'stdout.log').open('wb') as out, (a.out/'stderr.log').open('wb') as err:
            child=subprocess.Popen(a.command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            leader=child.pid; row=proc_rows().get(leader)
            if row is None: raise RuntimeError('Unable to establish child identity')
            start=row['start']; known={leader:start}; receipt.update(leader=leader,startticks=start)
            while True:
                if guard: guard.check()
                known,alive,outside=track(proc_rows(),leader,start,os.getpid(),known)
                if child.poll() is not None: receipt['reason']='exited'; break
                if interrupted: receipt['reason']='controller_signal'; break
                if time.monotonic()>=stop or time.time()>=pin['hard_stop_unix']-15: receipt['reason']='deadline'; break
                time.sleep(.05)
    except BaseException as e:
        receipt['error']=repr(e)
    finally:
        signals=[]
        if leader and start is not None:
            for sig,duration in ((signal.SIGTERM,2),(signal.SIGKILL,8)):
                until=time.monotonic()+duration
                while True:
                    known,alive,outside=track(proc_rows(),leader,start,os.getpid(),known)
                    if not alive: break
                    for row in alive:
                        current=proc_rows().get(row['pid'])
                        if current and current['start']==known[row['pid']]:
                            try: os.kill(row['pid'],sig); signals.append([row['pid'],row['start'],int(sig)])
                            except ProcessLookupError: pass
                    if time.monotonic()>=until: break
                    time.sleep(.05)
            child.poll()
            while True:
                try:
                    pid,status=os.waitpid(-1,os.WNOHANG)
                    if not pid: break
                except ChildProcessError: break
            known,alive,outside=track(proc_rows(),leader,start,os.getpid(),known)
            receipt.update(exit_code=child.returncode,owned=known,survivors=alive,cleanup_complete=not alive,signals=signals)
        if guard:
            try: receipt['memory_guard']=guard.close()
            except BaseException as e: receipt['memory_guard_error']=repr(e); receipt['cleanup_complete']=False
        receipt['finished_unix']=time.time()
        (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    return 0 if receipt.get('exit_code')==0 and receipt['cleanup_complete'] and receipt.get('reason')=='exited' and 'error' not in receipt else 1
if __name__=='__main__': sys.exit(main())



