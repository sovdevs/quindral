"""API entrypoint: POST /route {"query": "...", "eu_only": bool, "weights": {...}}
-> routing trace + NL explanation. GET / serves the web UI (index.html).
Stdlib http.server only, no framework dep installed yet — swap for
FastAPI/Flask when request volume or middleware needs (auth, validation,
docs) outgrow this. Run: python3 api.py --serve
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from router import route
from explain import explain

INDEX_HTML = Path(__file__).parent / "index.html"
LOGO_PNG = Path(__file__).parent / "logo.png"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._send_file(INDEX_HTML, "text/html; charset=utf-8")
        elif self.path == "/logo.png":
            self._send_file(LOGO_PNG, "image/png")
        else:
            self._send(404, {"error": "not found"})

    def _send_file(self, path: Path, content_type: str):
        if not path.exists():
            self._send(404, {"error": "not found"})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/route":
            self._send(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return

        query = body.get("query")
        if not query:
            self._send(400, {"error": "'query' is required"})
            return

        trace = route(
            query,
            weights=body.get("weights"),
            eu_only=body.get("eu_only", False),
            min_context=body.get("min_context", 0),
        )
        trace["explanation"] = explain(trace, eu_only=body.get("eu_only", False))
        self._send(200, trace)

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # ponytail: silence default stderr access log; add real logging if this ships


def serve(port: int = 8000):
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def _demo():
    import threading
    import urllib.request

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/route",
            data=json.dumps({"query": "what is the capital of france?"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        assert data["chosen"] is not None
        assert "explanation" in data

        bad_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/route",
            data=json.dumps({}).encode(),
            method="POST",
        )
        try:
            urllib.request.urlopen(bad_req)
            assert False, "should have raised HTTPError for missing query"
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        server.shutdown()

    print("api self-check: all cases passed")


if __name__ == "__main__":
    import sys

    if "--serve" in sys.argv:
        print("Listening on http://127.0.0.1:8000/route")
        serve()
    else:
        _demo()
