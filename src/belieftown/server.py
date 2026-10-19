"""Loopback-only development server. No arbitrary file serving or external assets."""
from __future__ import annotations
import atexit
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from importlib.resources import files
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlparse
from .studio import Studio

MAX_BODY=2_000_000


class StudioServer:
    def __init__(self,studio: Studio|None=None,port: int=0):
        self.studio=studio or Studio();self.token=secrets.token_urlsafe(24);owner=self
        class Handler(BaseHTTPRequestHandler):
            server_version='BeliefTown/1.0'
            def log_message(self,fmt,*args):pass
            def _headers_ok(self):
                host=self.headers.get('Host','')
                return host in (f'127.0.0.1:{owner.port}',f'localhost:{owner.port}')
            def send_data(self,status,data,content_type='application/json; charset=utf-8'):
                if isinstance(data,(dict,list)):data=json.dumps(data,allow_nan=False).encode()
                elif isinstance(data,str):data=data.encode()
                self.send_response(status);self.send_header('Content-Type',content_type)
                self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
            def do_GET(self):
                if not self._headers_ok():return self.send_data(403,{'error':'Loopback Host required'})
                path=urlparse(self.path).path
                assets={'/':'index.html','/app.js':'app.js','/styles.css':'styles.css'}
                if path in assets:
                    text=files('belieftown').joinpath('static',assets[path]).read_text(encoding="utf-8")
                    if path=='/':text=text.replace('__BOOTSTRAP__',json.dumps({'token':owner.token,'offline':False}))
                    mime='text/html; charset=utf-8' if path=='/' else ('application/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8')
                    return self.send_data(200,text,mime)
                if path=='/api/state':
                    if self.headers.get('X-BeliefTown-Token')!=owner.token:return self.send_data(403,{'error':'Session token required'})
                    return self.send_data(200,owner.studio.dispatch('state',{}))
                return self.send_data(404,{'error':'Not found'})
            def do_POST(self):
                if not self._headers_ok():return self.send_data(403,{'error':'Loopback Host required'})
                origin=self.headers.get('Origin')
                if origin and origin not in (f'http://127.0.0.1:{owner.port}',f'http://localhost:{owner.port}'):
                    return self.send_data(403,{'error':'Cross-origin requests are not allowed'})
                if self.headers.get('X-BeliefTown-Token')!=owner.token:return self.send_data(403,{'error':'Session token required'})
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=MAX_BODY:return self.send_data(413,{'error':'JSON request must be under 2 MB'})
                    if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send_data(415,{'error':'Use application/json'})
                    args=json.loads(self.rfile.read(length))
                    if not isinstance(args,dict):raise ValueError('Request must be a JSON object')
                    path=urlparse(self.path).path
                    if not path.startswith('/api/'):return self.send_data(404,{'error':'Not found'})
                    result=owner.studio.dispatch(path[5:],args)
                    return self.send_data(200,result)
                except (ValueError,KeyError,TypeError,IndexError,FileNotFoundError) as error:
                    return self.send_data(400,{'error':str(error)})
                except Exception:
                    # Do not leak filesystem details or stack traces into the UI.
                    import traceback;traceback.print_exc()
                    return self.send_data(500,{'error':'Internal error; see the local server terminal'})
        self.httpd=ThreadingHTTPServer(('127.0.0.1',port),Handler)
        self.port=self.httpd.server_address[1];self.thread=None;self.closed=False
    @property
    def url(self):return f'http://127.0.0.1:{self.port}/'
    def start(self):
        if self.thread is None:
            self.thread=threading.Thread(target=self.httpd.serve_forever,daemon=True,name='belieftown-local-server');self.thread.start()
        return self
    def close(self):
        if self.closed:return
        self.closed=True
        if self.thread:self.httpd.shutdown();self.thread.join(timeout=3)
        self.httpd.server_close()
    def __enter__(self):return self.start()
    def __exit__(self,*args):self.close()
    def _repr_html_(self):
        import html
        return f'<p><a href="{html.escape(self.url)}" target="_blank">Open BeliefTown in a separate local tab</a></p><iframe src="{html.escape(self.url)}" width="100%" height="1040" style="border:1px solid #ccc;border-radius:12px" title="BeliefTown counterfactual debugger"></iframe>'


def launch_notebook(scene=None,checkpoint_dir=None,port=0):
    server=StudioServer(Studio(scene,checkpoint_dir),port).start()
    atexit.register(server.close)
    return server
