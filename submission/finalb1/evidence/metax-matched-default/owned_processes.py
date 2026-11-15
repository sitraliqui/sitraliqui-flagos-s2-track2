"""Exact process-identity functions copied from reviewed TianShu guard; NO memory policy."""
import ctypes,os,signal,time
from pathlib import Path

def proc_rows(base=Path('/proc')):
    rows={}
    for item in base.iterdir():
        if not item.name.isdigit():continue
        try:
            parts=(item/'stat').read_text(encoding='utf-8').rsplit(') ',1)[1].split()
            rows[int(item.name)]=dict(pid=int(item.name),state=parts[0],ppid=int(parts[1]),
                pgid=int(parts[2]),session=int(parts[3]),start=int(parts[19]))
        except (OSError,IndexError,ValueError):continue
    return rows

def identity(pid):
    row=proc_rows()[pid]
    return (row['pid'],row['start'])

def subreaper():
    assert os.name=='posix' and Path('/proc').is_dir(),'Linux /proc required'
    libc=ctypes.CDLL(None,use_errno=True)
    assert libc.prctl(36,1,0,0,0)==0,'Cannot enable PR_SET_CHILD_SUBREAPER'

def track(rows,leader,start,observer,known):
    """Only this dedicated observer's new descendants; never infer from a port."""
    known=dict(known)
    if leader in rows and rows[leader]['start']==start:known[leader]=start
    changed=True
    while changed:
        changed=False
        for pid,row in rows.items():
            if row['start']<start or row['state']=='Z':continue
            parent_owned=row['ppid'] in known and row['ppid'] in rows and rows[row['ppid']]['start']==known[row['ppid']]
            # Subreaper adoption preserves visibility if an owned parent exits.
            adopted=row['ppid']==observer
            if (parent_owned or adopted) and pid not in known:
                known[pid]=row['start'];changed=True
    alive=[row for pid,row in rows.items() if known.get(pid)==row['start'] and row['state']!='Z']
    outside=[row for row in alive if row['pgid']!=leader or row['session']!=leader]
    group=[row for row in rows.values() if row['pgid']==leader and row['state']!='Z']
    assert all(known.get(row['pid'])==row['start'] and row['session']==leader for row in group),'Unknown process in owned PGID'
    return known,alive,outside

def cleanup_targets(rows,known,leader):
    """Pure selection helper: reused PID must never become a kill target."""
    return [row for pid,row in rows.items() if row['state']!='Z' and known.get(pid)==row['start']]

def cleanup(leader,start,observer,known,proc=None):
    for sig,seconds in ((signal.SIGTERM,2),(signal.SIGKILL,8)):
        rows=proc_rows();known,alive,outside=track(rows,leader,start,observer,known)
        if not alive:break
        group=[r for r in alive if r['pgid']==leader and r['session']==leader]
        if group:
            try:os.killpg(leader,sig)
            except ProcessLookupError:pass
        # Escaped descendants are killed individually only after exact identity
        # revalidation. Never signal their possibly unrelated new process group.
        for row in outside:
            current=proc_rows().get(row['pid'])
            if current and current['start']==known[row['pid']]:
                try:os.kill(row['pid'],sig)
                except ProcessLookupError:pass
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            rows=proc_rows();known,alive,outside=track(rows,leader,start,observer,known)
            if not alive:break
            time.sleep(.05)
    if proc is not None:
        try:proc.wait(timeout=2)
        except ChildProcessError:pass
    # Reap only children of this dedicated subreaper, which owns no other jobs.
    while True:
        try:
            pid,_=os.waitpid(-1,os.WNOHANG)
            if pid==0:break
        except ChildProcessError:break
    known,alive,outside=track(proc_rows(),leader,start,observer,known)
    assert not alive,'Owned process cleanup not confirmed'
    return known
