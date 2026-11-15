"""Read-only cgroup-v1/v2 host-memory observer and owned-process-group cleanup.

Protection is best effort, not a new kernel-enforced memory limit. No GPU imports.
"""
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import threading
import time

GiB = 1024 ** 3
SIGKILL = getattr(signal, 'SIGKILL', 9)  # Linux signal number for Windows synthetic CPU contracts.


class GuardStopped(RuntimeError):
    def __init__(self, reason, detail, snapshot):
        self.reason, self.detail, self.snapshot = reason, detail, snapshot
        super().__init__(reason + ': ' + detail)


def unescape_mount(value):
    return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), value)


def resolve_v2(cgroup_text, mountinfo_text):
    """Return (visible mount point, own cgroup directory); reject ambiguity/v1."""
    rows = [row.split(':', 2) for row in cgroup_text.splitlines() if row.strip()]
    if len(rows) != 1 or len(rows[0]) != 3 or rows[0][:2] != ['0', '']:
        raise ValueError('Only an unambiguous unified cgroup-v2 hierarchy is supported')
    cg = PurePosixPath(rows[0][2])
    if not cg.is_absolute() or '..' in cg.parts:
        raise ValueError('Invalid cgroup path')
    found = []
    for row in mountinfo_text.splitlines():
        halves = row.split(' - ', 1)
        if len(halves) != 2:
            continue
        before, after = halves[0].split(), halves[1].split()
        if not after or after[0] != 'cgroup2' or len(before) < 6:
            continue
        root, mount = PurePosixPath(unescape_mount(before[3])), PurePosixPath(unescape_mount(before[4]))
        if not root.is_absolute() or not mount.is_absolute() or '..' in root.parts + mount.parts:
            raise ValueError('Invalid cgroup2 mount path')
        if cg == PurePosixPath('/'):
            relative = PurePosixPath('.')  # cgroup namespace root exposed by this mount.
        elif cg.is_relative_to(root):
            relative = cg.relative_to(root)
        else:
            continue
        found.append((Path(str(mount)), Path(str(mount / relative))))
    found = list(dict.fromkeys(found))
    if len(found) != 1:
        raise ValueError('No unique cgroup2 mount resolves this process')
    mount, own = found[0]
    if not own.is_relative_to(mount):
        raise ValueError('Resolved cgroup escaped visible mount')
    return mount, own


def integer(text, *, unlimited=False):
    value = text.strip()
    if unlimited and value == 'max':
        return None
    if not value.isdecimal():
        raise ValueError('Invalid nonnegative cgroup integer')
    return int(value)


def counters(text):
    result = {}
    for line in text.splitlines():
        key, value = line.split()
        if key in result:
            raise ValueError('Duplicate memory counter')
        result[key] = integer(value)
    return result


class CgroupV2Reader:
    def __init__(self, proc_root=Path('/proc')):
        self.proc_root = Path(proc_root)
        self.mount, self.own = resolve_v2((self.proc_root/'self/cgroup').read_text(encoding='utf-8'),
                                          (self.proc_root/'self/mountinfo').read_text(encoding='utf-8'))
        # cgroupfs directories must not be redirected outside the visible mount.
        if not self.own.resolve().is_relative_to(self.mount.resolve()):
            raise ValueError('Resolved cgroup directory escaped visible mount')
        self.paths = []
        node = self.own
        while True:
            self.paths.append(node)
            if node == self.mount:
                break
            node = node.parent

    def __call__(self):
        scopes = []
        for node in self.paths:
            maximum = node/'memory.max'
            if not maximum.exists() and node == self.mount:
                # The true root may omit control files. It is not a finite limit.
                continue
            limit = integer(maximum.read_text(encoding='utf-8'), unlimited=True)
            current = integer((node/'memory.current').read_text(encoding='utf-8'))
            events = counters((node/'memory.events').read_text(encoding='utf-8'))
            if not {'oom', 'oom_kill'} <= set(events):
                raise ValueError('OOM counters unavailable')
            row = {'path': str(node), 'current_bytes': current, 'limit_bytes': limit,
                   'events': events, 'stat': counters((node/'memory.stat').read_text(encoding='utf-8'))}
            for filename, key, unlimited in [('memory.swap.current', 'swap_current_bytes', False),
                                               ('memory.swap.max', 'swap_max_bytes', True)]:
                path = node/filename
                if path.exists():
                    row[key] = integer(path.read_text(encoding='utf-8'), unlimited=unlimited)
            scopes.append(row)
        finite = [row['limit_bytes'] for row in scopes if row['limit_bytes'] is not None]
        if not finite:
            raise ValueError('No visible finite cgroup-v2 host-memory limit')
        return {'version': 2, 'own_path': str(self.own), 'effective_limit_bytes': min(finite), 'scopes': scopes}


def resolve_v1(cgroup_text,mountinfo_text):
    rows=[line.split(':',2) for line in cgroup_text.splitlines() if line.strip()]
    memory=[row[2] for row in rows if len(row)==3 and 'memory' in row[1].split(',')]
    if len(memory)!=1:raise ValueError('No unique cgroup-v1 memory hierarchy')
    cg=PurePosixPath(memory[0]);found=[]
    if not cg.is_absolute() or '..' in cg.parts:raise ValueError('Invalid memory cgroup path')
    for line in mountinfo_text.splitlines():
        before,sep,after=line.partition(' - ')
        if not sep:continue
        a,b=before.split(),after.split()
        if len(a)<6 or len(b)<3 or b[0]!='cgroup' or 'memory' not in b[2].split(','):continue
        root,mount=PurePosixPath(unescape_mount(a[3])),PurePosixPath(unescape_mount(a[4]))
        if not root.is_absolute() or not mount.is_absolute() or '..' in root.parts+mount.parts:raise ValueError('Invalid memory mount')
        if cg==PurePosixPath('/'):relative=PurePosixPath('.')
        elif cg.is_relative_to(root):relative=cg.relative_to(root)
        else:continue
        found.append((Path(str(mount)),Path(str(mount/relative))))
    found=list(dict.fromkeys(found))
    if len(found)!=1:raise ValueError('No unique visible cgroup-v1 memory mount')
    return found[0]


class CgroupV1Reader:
    def __init__(self,proc_root=Path('/proc')):
        self.proc_root=Path(proc_root)
        self.mount,self.own=resolve_v1((self.proc_root/'self/cgroup').read_text(encoding='utf-8'),
                                      (self.proc_root/'self/mountinfo').read_text(encoding='utf-8'))
        if not self.own.resolve().is_relative_to(self.mount.resolve()):raise ValueError('Memory cgroup escaped mount')
        self.paths=[];node=self.own
        while True:
            self.paths.append(node)
            if node==self.mount:break
            node=node.parent

    def __call__(self):
        scopes=[]
        for node in self.paths:
            limit=integer((node/'memory.limit_in_bytes').read_text(encoding='utf-8'))
            # v1 represents unlimited by a page-rounded near-LONG_MAX value.
            limit=None if limit>=2**60 else limit
            oom=counters((node/'memory.oom_control').read_text(encoding='utf-8'))
            if not {'under_oom','oom_kill'}<=set(oom):raise ValueError('v1 OOM evidence unavailable')
            events={'under_oom':oom['under_oom'],'oom_kill':oom['oom_kill'],
                    'failcnt':integer((node/'memory.failcnt').read_text(encoding='utf-8'))}
            scopes.append(dict(path=str(node),limit_bytes=limit,
                current_bytes=integer((node/'memory.usage_in_bytes').read_text(encoding='utf-8')),
                events=events,stat=counters((node/'memory.stat').read_text(encoding='utf-8'))))
        finite=[row['limit_bytes'] for row in scopes if row['limit_bytes'] is not None]
        if not finite:raise ValueError('No visible finite v1 memory limit')
        return dict(version=1,own_path=str(self.own),effective_limit_bytes=min(finite),scopes=scopes)


def cgroup_reader():
    text=Path('/proc/self/cgroup').read_text(encoding='utf-8')
    return CgroupV1Reader() if any(len(row.split(':',2))==3 and 'memory' in row.split(':',2)[1].split(',') for row in text.splitlines()) else CgroupV2Reader()


class ProcProcesses:
    def identity(self, pid):
        try:
            text = Path('/proc', str(pid), 'stat').read_text(encoding='utf-8')
            fields = text[text.rfind(')')+2:].split()
            return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]), 'pgid': int(fields[2]),
                    'sid': int(fields[3]), 'start': int(fields[19])}
        except FileNotFoundError:
            return None

    def scan_group(self, pgid):
        members = []
        for path in Path('/proc').iterdir():
            if path.name.isdecimal():
                try:
                    row = self.identity(int(path.name))
                    if row and row['pgid'] == pgid:
                        members.append(row)
                except (FileNotFoundError, ProcessLookupError):
                    continue
        return members

    def spawn(self, command, **kwargs):
        return subprocess.Popen(command, **kwargs)

    def signal_group(self, pgid, sig):
        os.killpg(pgid, sig)


class MemoryGuard:
    def __init__(self, output_dir, *, reader=None, process_api=None, _interval=.25,
                 _term_grace_seconds=2.0, _ordinary_log_interval=1.0, _max_log_rows=4096):
        self.output = Path(output_dir)
        self._reader, self._ops = reader, process_api or ProcProcesses()
        self._interval, self._grace = _interval, _term_grace_seconds
        self._log_interval, self._log_cap = _ordinary_log_interval, _max_log_rows
        if not 0 < self._interval <= .25 or not 0 <= self._grace <= 2 or self._log_cap < 16:
            raise ValueError('Unsafe watchdog bounds')
        self._lock = threading.RLock()
        self._halt, self._stopped = threading.Event(), threading.Event()
        self._thread = None
        self._started, self._closed = False, False
        self._reason = self._detail = None
        self._baseline = self._latest = self._trigger = None
        self._groups = []
        self._log_rows = self._log_dropped = 0
        self._last_log = 0.0
        self._fatal = False
        self._receipt = None

    def _read(self):
        sample = self._reader()
        if not isinstance(sample, dict) or sample.get('version') not in (1,2) or not isinstance(sample.get('scopes'), list) or not 1 <= len(sample['scopes']) <= 256:
            raise ValueError('Cgroup sample unavailable')
        limits = []
        for row in sample['scopes']:
            if not isinstance(row, dict) or not isinstance(row.get('path'), str) or not isinstance(row.get('events'), dict):
                raise ValueError('Invalid memory scope')
            if type(row.get('current_bytes')) is not int or row['current_bytes'] < 0:
                raise ValueError('Invalid current usage')
            limit = row.get('limit_bytes')
            if limit is not None:
                if type(limit) is not int or limit <= 8*GiB:
                    raise ValueError('Finite limit cannot retain requested 8 GiB margin')
                limits.append(limit)
            required={'under_oom','oom_kill','failcnt'} if sample['version']==1 else {'oom','oom_kill'}
            if not required <= set(row.get('events', {})) or any(type(v) is not int or v < 0 for v in row['events'].values()):
                raise ValueError('OOM counters unavailable/invalid')
        if not limits or sample.get('effective_limit_bytes') != min(limits):
            raise ValueError('Finite parent-limit accounting differs')
        sample = dict(sample, monotonic=time.monotonic(), unix=time.time())
        sample['soft_threshold_bytes'] = sample['effective_limit_bytes']-8*GiB
        sample['critical_threshold_bytes'] = sample['effective_limit_bytes']-4*GiB
        return sample

    def _log(self, kind, payload, *, trigger=False):
        if self._log_rows >= self._log_cap - 8 and not trigger:
            self._log_dropped += 1
            return
        if self._log_rows >= self._log_cap:
            self._log_dropped += 1
            return
        row = {'kind': kind, 'unix': time.time(), **payload}
        with (self.output/'memory-guard.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
        self._log_rows += 1

    def _evaluate(self, sample):
        old = {row['path']: row for row in self._baseline['scopes']}
        if sample['version']!=self._baseline['version']:
            return 'host_memory_monitor_blind','cgroup_version_changed',True
        if set(old) != {row['path'] for row in sample['scopes']}:
            return 'host_memory_monitor_blind', 'cgroup_scope_changed', True
        for row in sample['scopes']:
            previous = old[row['path']]
            if row['limit_bytes']!=previous['limit_bytes']:
                return 'host_memory_monitor_blind','cgroup_limit_changed',True
            if sample['version']==1 and row['events']['under_oom']:
                return 'host_memory_oom_event','under_oom',True
            for key in (('oom_kill','failcnt') if sample['version']==1 else ('oom', 'oom_kill', 'oom_group_kill')):
                before, after = previous['events'].get(key, 0), row['events'].get(key, 0)
                if after < before:
                    return 'host_memory_monitor_blind', 'oom_counter_reset', True
                if after > before:
                    return ('host_memory_allocation_failure' if key=='failcnt' else 'host_memory_oom_event'), key+'_increment', True
        for row in sample['scopes']:
            limit = row['limit_bytes']
            if limit is not None and row['current_bytes'] >= limit-4*GiB:
                return 'host_memory_guard_stop', 'critical_headroom_below_4GiB', True
        for row in sample['scopes']:
            limit = row['limit_bytes']
            if limit is not None and row['current_bytes'] >= limit-8*GiB:
                return 'host_memory_guard_stop', 'soft_headroom_below_8GiB', False
        return None

    def _identity(self, group):
        """Fresh witness proves this group's continuity; never signal by stale PID."""
        try:
            group['child'].poll()  # reap the owned leader when it has exited.
            members = self._ops.scan_group(group['pgid'])
            if not members:
                group['state'] = 'gone'
                return 'gone'
            leader = next((row for row in members if row['pid'] == group['pid']), None)
            if leader is not None and (leader['start'] != group['start'] or leader['sid'] != group['pgid']):
                group['state'] = 'cleanup_unknown'
                return 'unknown'
            witness = any(row['sid'] == group['pgid'] and group['witnesses'].get(row['pid']) == row['start'] for row in members)
            if not witness or any(row['sid'] != group['pgid'] for row in members):
                group['state'] = 'cleanup_unknown'
                return 'unknown'
            group['witnesses'].update({row['pid']: row['start'] for row in members})
            group['state'] = 'alive_verified'
            return 'verified'
        except Exception:
            group['state'] = 'cleanup_unknown'
            return 'unknown'

    def _signal(self, group, sig):
        if self._identity(group) != 'verified':
            return
        try:
            self._ops.signal_group(group['pgid'], sig)
            group['signals'].append({'signal': int(sig), 'monotonic': time.monotonic()})
        except ProcessLookupError:
            group['state'] = 'gone'
        except Exception as error:
            group['state'] = 'cleanup_unknown'
            group['signal_error'] = type(error).__name__

    def _terminate(self, *, force):
        now = time.monotonic()
        for group in self._groups:
            if force:
                if not group['kill_sent']:
                    self._signal(group, SIGKILL)
                    group['kill_sent'] = True
            elif group['term_at'] is None:
                self._signal(group, signal.SIGTERM)
                group['term_at'] = now
            elif now-group['term_at'] >= self._grace and not group['kill_sent']:
                self._signal(group, SIGKILL)
                group['kill_sent'] = True

    def _trip(self, reason, detail, sample, fatal):
        first, escalation = not self._stopped.is_set(), fatal and not self._fatal
        if first:
            self._reason, self._detail, self._trigger = reason, detail, sample
            self._stopped.set()
        if fatal:
            self._fatal = True
            if not first:
                self._detail = detail
        # Kill first, then do diagnostic I/O: log errors must not suppress cleanup.
        self._terminate(force=self._fatal)
        if first or escalation:
            try:
                self._log('stop' if first else 'escalation', {'reason': reason, 'detail': detail, 'sample': sample}, trigger=True)
            except Exception:
                self._log_dropped += 1

    def start(self):
        with self._lock:
            if self._started or self._closed:
                raise RuntimeError('Guard start is one-shot')
            self.output.mkdir(parents=True, exist_ok=True)
            if (self.output/'memory-guard.jsonl').exists() or (self.output/'memory-guard-receipt.json').exists():
                raise RuntimeError('Guard output already exists')
            self._started = True
            try:
                if isinstance(self._ops, ProcProcesses) and os.name != 'posix':
                    raise RuntimeError('Real process groups require Linux; only injected CPU process contracts are portable')
                if self._reader is None:
                    self._reader = cgroup_reader()
                self._baseline = self._latest = self._read()
                event = self._evaluate(self._baseline)
                if event:
                    self._trip(event[0], event[1], self._baseline, event[2])
                else:
                    self._log('baseline', {'sample': self._baseline})
            except Exception as error:
                self._trip('host_memory_monitor_blind', 'initial_'+type(error).__name__, None, True)
            self._thread = threading.Thread(target=self._watch, name='host-memory-guard', daemon=True)
            self._thread.start()
            self.check()
            return self

    def _watch(self):
        while not self._halt.wait(self._interval):
            with self._lock:
                try:
                    sample = self._read()
                    self._latest = sample
                    event = self._evaluate(sample)
                    if event:
                        self._trip(event[0], event[1], sample, event[2])
                    if time.monotonic()-self._last_log >= self._log_interval:
                        self._log('sample', {'sample': sample})
                        self._last_log = time.monotonic()
                except Exception as error:
                    self._trip('host_memory_monitor_blind', 'sample_'+type(error).__name__, self._latest, True)
                if self._stopped.is_set():
                    self._terminate(force=self._fatal)
                else:
                    for group in self._groups:
                        self._identity(group)

    def check(self):
        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError('Guard is not active')
            if self._thread is not None and not self._thread.is_alive() and not self._halt.is_set():
                self._trip('host_memory_monitor_blind', 'watchdog_thread_exited', self._latest, True)
            if self._stopped.is_set():
                raise GuardStopped(self._reason, self._detail, self.snapshot())

    def spawn(self, command, *, label, **kwargs):
        with self._lock:
            self.check()
            if not isinstance(label, str) or not label or any(group['label'] == label for group in self._groups):
                raise ValueError('Unique nonempty owned-group label required')
            if kwargs.get('start_new_session', True) is not True or kwargs.get('preexec_fn') is not None or kwargs.get('process_group') is not None or kwargs.get('shell', False):
                raise ValueError('Owned groups require direct exec and a new session')
            # Fresh sample under the same lock closes the Event-check/Popen registration race.
            try:
                sample = self._read();self._latest = sample
                event = self._evaluate(sample)
                if event:
                    self._trip(event[0], event[1], sample, event[2])
            except Exception as error:
                self._trip('host_memory_monitor_blind', 'spawn_'+type(error).__name__, self._latest, True)
            self.check()
            kwargs['start_new_session'] = True
            child = self._ops.spawn(command, **kwargs)
            try:
                identity = self._ops.identity(child.pid)
            except Exception:
                identity = None
            group = {'child': child, 'label': label, 'pid': child.pid, 'pgid': child.pid,
                     'start': identity['start'] if identity else None, 'witnesses': {},
                     'term_at': None, 'kill_sent': False, 'signals': [], 'state': 'cleanup_unknown'}
            self._groups.append(group)
            if identity is not None and identity['pid'] == child.pid and identity['pgid'] == child.pid and identity['sid'] == child.pid:
                group['witnesses'][child.pid] = identity['start'];group['state'] = 'alive_verified'
            else:
                # Never guess a process group when registration identity is unavailable.
                self._trip('host_memory_monitor_blind', 'spawn_identity_unverified', self._latest, True)
                self.check()
            return child

    def snapshot(self):
        with self._lock:
            groups = []
            for group in self._groups:
                self._identity(group)
                groups.append({key: group.get(key) for key in ('label', 'pid', 'pgid', 'start', 'state', 'signals', 'signal_error')})
            return {'started': self._started, 'closed': self._closed, 'stopped': self._stopped.is_set(),
                    'stop_reason': self._reason, 'stop_detail': self._detail, 'fatal': self._fatal,
                    'baseline': self._baseline, 'latest': self._latest, 'trigger': self._trigger,
                    'groups': groups, 'cleanup_unknown': any(row['state']=='cleanup_unknown' for row in groups),
                    'owned_groups_gone': all(row['state']=='gone' for row in groups),
                    'log_rows': self._log_rows, 'log_dropped': self._log_dropped}

    def stop_all(self):
        with self._lock:
            if not self._started:
                return self.snapshot()
            if self._thread is not None and not self._thread.is_alive() and not self._halt.is_set():
                self._trip('host_memory_monitor_blind', 'watchdog_thread_exited_before_cleanup', self._latest, True)
            self._terminate(force=self._fatal)
        # Watchdog remains active through the normal TERM grace and cleanup.
        end = time.monotonic()+self._grace+.5
        while time.monotonic() < end:
            with self._lock:
                self._terminate(force=self._fatal)
                view = self.snapshot()
                if view['owned_groups_gone'] or view['cleanup_unknown']:
                    return view
            time.sleep(min(.02, self._interval))
        with self._lock:
            self._terminate(force=True)
            return self.snapshot()

    def close(self):
        if self._closed:
            return self._receipt
        if not self._started:
            self._closed = True
            self._receipt = {'started': False, 'closed': True, 'close_pending': False, 'thread_stopped': True,
                             'owned_groups_gone': True, 'cleanup_unknown': False, 'groups': [], 'stop_reason': None}
            return self._receipt
        view = self.stop_all()
        complete = view['owned_groups_gone']
        if complete:
            self._halt.set()
            if self._thread and self._thread is not threading.current_thread():
                self._thread.join(timeout=1.0)
        with self._lock:
            receipt = self.snapshot()
            receipt.update(closed=complete, close_pending=not complete,
                           thread_stopped=not self._thread or not self._thread.is_alive(),
                           policy={'cgroup_version': self._baseline['version'] if self._baseline else None, 'soft_reserve_bytes': 8*GiB, 'critical_reserve_bytes': 4*GiB,
                                   'interval_seconds': self._interval, 'term_grace_seconds': self._grace},
                           validation='Host-memory protection only; no GPU validation; polling is not a hard memory quota')
            self._closed = complete;self._receipt = receipt
            temp = self.output/'memory-guard-receipt.json.tmp'
            temp.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            temp.replace(self.output/'memory-guard-receipt.json')
            return receipt
