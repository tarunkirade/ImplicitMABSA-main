"""OPTIONAL web app for ImplicitMABSA.  The project works without it (see run.py); this is just a friendlier face.

    python run.py ui            (or)    python ui/server.py

The server is a thin layer over src/implicitmabsa/api.py - no business logic lives here.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from implicitmabsa import api  # noqa: E402
from implicitmabsa.lexicon import EXPLICIT_TERMS  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
STATIC = os.path.join(ROOT, "ui", "static")
REAL_DIR = os.path.join(ROOT, "data", "real")
PAGE = 24

_lock = threading.Lock()
_state = {"an": None, "err": None, "clip": None, "clip_err": None}
_cache = {}


def jread(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def rows():
    if "rows" not in _cache:
        _cache["rows"] = api.load_rows()
        _cache["by_id"] = {r["id"]: r for r in _cache["rows"]}
    return _cache["rows"]


def real():
    if "real" not in _cache:
        d = jread(os.path.join(REAL_DIR, "samples.json"), {"samples": []})
        _cache["real"] = d["samples"]
        _cache["real_by_id"] = {s["id"]: s for s in d["samples"]}
    return _cache["real"]


def analyzer():
    if _state["an"] is None and not _state["err"]:
        try:
            _state["an"] = api.Analyzer()
        except Exception as e:  # noqa: BLE001
            _state["err"] = f"{type(e).__name__}: {e}"
    return _state["an"]


def clip_engine():
    """Real-photo engine (pretrained CLIP).  Loaded on first use; the first time it downloads ~350 MB."""
    if _state["clip"] is None:
        try:
            from implicitmabsa.clip_engine import ClipAnalyzer

            _state["clip"] = ClipAnalyzer()
            _state["clip_err"] = None
        except ImportError:
            _state["clip_err"] = "Real-photo mode needs two extra packages. Run:  pip install open_clip_torch vaderSentiment  (run.bat does this for you)"
        except Exception as e:  # noqa: BLE001
            _state["clip_err"] = f"Could not load the real-photo model ({type(e).__name__}: {e}). It needs an internet connection the first time."
    return _state["clip"]


def clip_installed():
    try:
        import open_clip, vaderSentiment  # noqa: F401, E401
        return True
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ API
def api_status():
    try:
        import torch  # noqa: F401
        torch_ok = True
    except Exception:  # noqa: BLE001
        torch_ok = False
    return {"torch": torch_ok, "checkpoint": os.path.exists(os.path.join(api.CKPT, "cross_seed0.pt")),
            "synthetic_rows": len(rows()), "images_full": api.count_images(), "real_samples": len(real()),
            "model_error": _state["err"], "clip_installed": clip_installed(), "clip_loaded": _state["clip"] is not None}


def api_results():
    return {"summary": jread(os.path.join(RESULTS, "summary.json"), {}),
            "real": jread(os.path.join(RESULTS, "real", "real_results.json"), {}),
            "stats": jread(os.path.join(api.DATA, "dataset_stats.json"), {}),
            "clip": jread(os.path.join(RESULTS, "clip_eval.json"), {}),
            "sanity": jread(os.path.join(RESULTS, "real_photo_sanity.json"), {}),
            "heads": jread(os.path.join(RESULTS, "clip_heads.json"), {})}


def api_dataset(q):
    g = lambda k: (q.get(k, [""])[0] or "").strip()  # noqa: E731
    text = g("q").lower()
    out = [r for r in rows() if (not g("split") or r["split"] == g("split")) and (not g("aspect") or r["aspect"] == g("aspect"))
           and (not g("sentiment") or r["sentiment"] == g("sentiment")) and (not text or text in r["text"].lower())]
    page = max(int(g("page") or 1), 1)
    items = [{"id": r["id"], "split": r["split"], "text": r["text"], "aspect": r["aspect"], "sentiment": r["sentiment"],
              "aspect_src": r["aspect_src"], "sent_src": r["sent_src"], "conflict": r["conflict"] == "1",
              "has_image": api.image_path(r) is not None} for r in out[(page - 1) * PAGE: page * PAGE]]
    return {"total": len(out), "page": page, "pages": max((len(out) + PAGE - 1) // PAGE, 1), "items": items}


def api_real(q):
    g = lambda k: (q.get(k, [""])[0] or "").strip()  # noqa: E731
    text = g("q").lower()
    out = [s for s in real() if (not g("year") or s["year"] == g("year")) and (not g("sentiment") or s["label"] == g("sentiment"))
           and (not text or text in s["text"].lower())]
    page = max(int(g("page") or 1), 1)
    return {"total": len(out), "page": page, "pages": max((len(out) + PAGE - 1) // PAGE, 1),
            "items": out[(page - 1) * PAGE: page * PAGE]}


def explicit_terms_in(text):
    low = text.lower()
    return sorted({t for t in EXPLICIT_TERMS if re.search(r"\b" + re.escape(t.lower()), low)})


def api_predict(body):
    from PIL import Image

    engine = body.get("engine", "trained")
    an = clip_engine() if engine == "clip" else analyzer()
    if an is None:
        return {"error": _state["clip_err"] if engine == "clip" else (_state["err"] or "Model unavailable")}, 500
    text, truth = (body.get("text") or "").strip(), None
    if body.get("id"):
        rows()
        row = _cache["by_id"].get(body["id"])
        if not row:
            return {"error": "Unknown sample"}, 404
        path = api.image_path(row)
        if not path:
            return {"error": "That image has not been created yet. Run  python run.py setup  once."}, 404
        image, text = Image.open(path), text or row["text"]
        truth = {"aspect": row["aspect"], "sentiment": row["sentiment"]}
    else:
        m = re.match(r"data:image/[\w.+-]+;base64,(.*)$", body.get("image_data_url") or "", re.S)
        if not m:
            return {"error": "Choose an image first."}, 400
        raw = base64.b64decode(m.group(1))
        if len(raw) > 12 * 1024 * 1024:
            return {"error": "Image is larger than 12 MB"}, 400
        image = Image.open(io.BytesIO(raw))
    if not text:
        return {"error": "Type a caption first."}, 400
    import time

    with _lock:
        t0 = time.perf_counter()
        res = an.predict(text[:400], image)
        decide_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        ex = an.explain(text[:400], image) if body.get("explain", True) else None
        explain_ms = (time.perf_counter() - t1) * 1000
    res.update(caption=text, truth=truth, explicit_terms=explicit_terms_in(text), engine=engine, explain=ex,
               timing={"decide_ms": decide_ms, "explain_ms": explain_ms})
    return res, 200


def api_evaluate(body):
    an = analyzer()
    if an is None:
        return {"error": _state["err"] or "Model unavailable"}, 500
    split, n = body.get("split", "test_iid"), max(20, min(int(body.get("n", 300)), 800))
    sel = [r for r in rows() if r["split"] == split and api.image_path(r)][:n]
    if len(sel) < 20:
        return {"error": f"Only {len(sel)} images of '{split}' exist. Run  python run.py setup  once."}, 400
    with _lock:
        out = api.evaluate(an, sel)
    out["split"] = split
    return out, 200


# ------------------------------------------------------------------ HTTP
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "application/javascript", ".png": "image/png",
        ".jpg": "image/jpeg", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith(("application/json", "text/html")) else "max-age=600")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        if not path or not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        with open(path, "rb") as f:
            self._send(200, f.read(), MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream"))

    def do_GET(self):
        u, q = urlparse(self.path), parse_qs(urlparse(self.path).query)
        try:
            if u.path in ("/", "/index.html"):
                return self._file(os.path.join(STATIC, "index.html"))
            if u.path == "/api/status":
                return self._send(200, api_status())
            if u.path == "/api/results":
                return self._send(200, api_results())
            if u.path == "/api/dataset":
                return self._send(200, api_dataset(q))
            if u.path == "/api/real":
                return self._send(200, api_real(q))
            if u.path == "/api/image":
                rows()
                row = _cache["by_id"].get((q.get("id") or [""])[0])
                return self._file(api.image_path(row) if row else None)
            if u.path == "/api/real_image":
                sid = re.sub(r"[^\w]", "", (q.get("id") or [""])[0])
                return self._file(os.path.join(REAL_DIR, "thumbs", sid + ".jpg"))
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20 * 1024 * 1024:
                return self._send(413, {"error": "Request too large"})
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/quit":
                self._send(200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            fn = {"/api/predict": api_predict, "/api/evaluate": api_evaluate}.get(u.path)
            out, code = fn(body) if fn else ({"error": "not found"}, 404)
            self._send(code, out)
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    rows(), real()
    url = f"http://{a.host}:{a.port}"
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"\n  ImplicitMABSA app running at {url}\n  To stop: click Quit in the app (top right), or press Ctrl+C.\n")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    threading.Thread(target=analyzer, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
