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
            self.poll()
            self.stop.wait(self.interval)

    def state(self):
        with self.lock:
            return json.dumps(self.value, allow_nan=False).encode()


def handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/', '/index.html'):
                page = 'training.html' if (monitor.value.get('snapshot') or {}).get('mode') == 'training' else 'index.html'
                body, mime = (HERE / page).read_bytes(), 'text/html; charset=utf-8'
            elif self.path == '/api/state':
                body, mime = monitor.state(), 'application/json'
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='h100-private')
    ap.add_argument('--source', choices=('ssh', 'local'), default='ssh')
    ap.add_argument('--python', default=None)
    ap.add_argument('--campaign', default='/home/taiga-support/mtg-ml-256-opt/campaigns/screens-20261008-dedicated')
    ap.add_argument('--remote-script', default='/home/taiga-support/mtg-ml-256-opt/dashboard_snapshot.py')
    ap.add_argument('--port', type=int, default=8767)
    ap.add_argument('--interval', type=int, default=15)
    ap.add_argument('--cache', default='.context/training-dashboard/state.json')
    args = ap.parse_args()
    if args.interval < 5:
        ap.error('Refresh interval must be at least five seconds')
    script = str(HERE / 'snapshot.py') if args.source == 'local' else args.remote_script
    interpreter = args.python or (sys.executable if args.source == 'local' else 'python3')
    monitor = Monitor(args.host, script, args.campaign, args.cache, args.interval, args.source, interpreter)
    monitor.poll()
    worker = threading.Thread(target=monitor.run, daemon=True)
    worker.start()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(monitor))
    print(f'Training dashboard: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        monitor.stop.set()
        server.server_close()
