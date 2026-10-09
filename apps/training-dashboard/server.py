"""Loopback dashboard; polls a read-only snapshot over the user's SSH connection."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import threading
import time
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


sys.path.insert(0, str(HERE))
import tracker  # noqa: E402


class TrackerMonitor:
    """Local, discovery-based monitor: finds campaigns and run directories itself on every poll."""

    def __init__(self, roots, explicit, ledger, interval, recent_hours):
        self.roots, self.explicit, self.ledger = roots, explicit, Path(ledger)
        self.interval, self.recent_hours = interval, recent_hours
        self.reader, self.stop, self.lock = tracker.RunReader(), threading.Event(), threading.Lock()
        self.value = {'snapshot': None, 'error': None, 'last_attempt': None, 'refresh_seconds': interval}
        self._ledger = (None, [])

    def poll(self):
        try:
            data = tracker.build_state(self.roots, self.explicit, self.reader, recent_hours=self.recent_hours)
            with self.lock:
                self.value.update(snapshot=data, error=None, last_attempt=time.time())
        except Exception as exc:  # keep serving the last good state
            with self.lock:
                self.value.update(error=f'{type(exc).__name__}: {exc}'[:500], last_attempt=time.time())

    def run(self):
        while not self.stop.is_set():
            started = time.monotonic()
            self.poll()
            self.stop.wait(max(0, self.interval - (time.monotonic() - started)))

    def state(self):
        with self.lock:
            return json.dumps(tracker.clean(self.value), allow_nan=False).encode()

    def history(self):
        try:
            mtime = self.ledger.stat().st_mtime
        except OSError:
            mtime = None
        if mtime != self._ledger[0]:
            self._ledger = (mtime, tracker.load_ledger(self.ledger))
        return json.dumps(tracker.clean(dict(ledger=str(self.ledger), runs=self._ledger[1])), allow_nan=False).encode()

    def report(self, name):
        """morning-report of a discovered campaign (matched by name, never by path)."""
        snap = self.value.get('snapshot') or {}
        for c in snap.get('campaigns', []):
            if name in (c['name'] + '.md', c['name'] + '.json'):
                f = Path(c['path']) / ('morning-report.' + name.rsplit('.', 1)[1])
                if f.is_file():
                    return f.read_bytes(), ('application/json' if f.suffix == '.json' else 'text/plain; charset=utf-8')
        return None, None


class Monitor:
    def __init__(self, host, script, campaign, cache, interval, source="ssh", python="python3"):
        self.host, self.script, self.campaign = host, script, campaign
        self.source, self.python = source, python
        self.cache, self.interval = Path(cache), interval
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.value = {'snapshot': None, 'error': None, 'last_attempt': None, 'refresh_seconds': interval}

    def poll(self):
        command = [self.python, self.script, '--campaign', self.campaign]
        if self.source == 'ssh':
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=5', self.host, shlex.join(command)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
            data = json.loads(result.stdout)
            if data.get('schema_version') != 1:
                raise ValueError('Unsupported dashboard snapshot')
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, allow_nan=False))
            temporary.replace(self.cache)
            with self.lock:
                self.value.update(snapshot=data, error=None, last_attempt=time.time())
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            with self.lock:
                self.value.update(error=str(exc)[:500], last_attempt=time.time())

    def run(self):
        while not self.stop.is_set():
            started = time.monotonic()
            self.poll()
            self.stop.wait(max(0, self.interval - (time.monotonic() - started)))

    def state(self):
        with self.lock:
            return json.dumps(self.value, allow_nan=False).encode()


def handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.path = self.path.split('?', 1)[0]
            tracking = isinstance(monitor, TrackerMonitor)
            if self.path in ('/', '/index.html'):
                page = 'tracker.html' if tracking else 'index.html'
                body, mime = (HERE / page).read_bytes(), 'text/html; charset=utf-8'
            elif tracking and self.path == '/api/history':
                body, mime = monitor.history(), 'application/json'
            elif tracking and self.path.startswith('/report/'):
                body, mime = monitor.report(self.path.removeprefix('/report/'))
                if body is None:
                    self.send_error(404)
                    return
            elif self.path == '/api/state':
                body, mime = monitor.state(), 'application/json'
            elif not tracking and self.path in ('/morning-report.json', '/morning-report.md'):
                path = Path(monitor.campaign) / self.path.removeprefix('/')
                if monitor.source != 'local' or not path.is_file():
                    self.send_error(404)
                    return
                body = path.read_bytes()
                mime = 'application/json' if path.suffix == '.json' else 'text/plain; charset=utf-8'
            elif self.path == '/healthz':
                body, mime = json.dumps({'ok': True, 'last_attempt': monitor.value['last_attempt'], 'error': monitor.value['error']}).encode(), 'application/json'
            elif self.path == '/favicon.ico':
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host')
    ap.add_argument('--source', choices=('ssh', 'local'), default='local')
    ap.add_argument('--python', default=None)
    ap.add_argument('--campaign', action='append', default=[], help='explicit campaign directory (repeatable)')
    ap.add_argument('--root', action='append', default=None,
                    help='glob of campaign / run directories to discover (repeatable; default %s)' % ' '.join(tracker.DEFAULT_ROOTS))
    ap.add_argument('--ledger', default=str(HERE.parent.parent / 'docs' / 'experiments' / 'ledger.jsonl'))
    ap.add_argument('--recent-hours', type=float, default=72, help='hide idle runs older than this')
    ap.add_argument('--remote-script')
    ap.add_argument('--port', type=int, default=8767)
    ap.add_argument('--interval', type=int, default=15)
    ap.add_argument('--cache', default='.context/training-dashboard/state.json')
    args = ap.parse_args()
    if args.interval < 5:
        ap.error('Refresh interval must be at least five seconds')
    if args.source == 'ssh':
        if not (args.host and args.remote_script and len(args.campaign) == 1):
            ap.error('SSH mode requires --host, --remote-script and exactly one --campaign')
        monitor = Monitor(args.host, args.remote_script, args.campaign[0], args.cache, args.interval, 'ssh', args.python or 'python3')
    else:
        monitor = TrackerMonitor(args.root if args.root is not None else list(tracker.DEFAULT_ROOTS), args.campaign,
                                 args.ledger, args.interval, args.recent_hours)
    monitor.poll()
    worker = threading.Thread(target=monitor.run, daemon=True)
    worker.start()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(monitor))
    print(f'Training tracker: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        monitor.stop.set()
        server.server_close()


if __name__ == '__main__':
    main()
