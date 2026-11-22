"""The project's public, UI-free interface.  Everything the web app and the terminal commands do goes through here,
so the whole project can be used from plain Python:

    from implicitmabsa.api import Analyzer
    a = Analyzer()                                   # loads checkpoints/cross_seed0.pt
    r = a.predict("Not what I expected!", "data/sample_images/test_iid_00001.png")
    r["both"]["aspect"], r["both"]["sentiment"]      # -> ('packaging', 'negative')
    r["text"], r["image"]                            # same model with the image / the text switched off

    from implicitmabsa.api import load_rows, evaluate
    evaluate(a, load_rows(split="test_iid"))         # accuracy / precision / recall / F1 per input mode
"""
import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")

from .lexicon import ASPECTS, SENTIMENTS  # noqa: E402

MODES = (("both", None), ("text", "image"), ("image", "text"))  # (name, input that is switched OFF)


# ------------------------------------------------------------------ dataset helpers
def load_rows(split=None, data_dir=DATA):
    """Rows of data/metadata.csv (optionally one split)."""
    path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split in (None, r["split"])]


def image_path(row, data_dir=DATA):
    """data/images/<file>, falling back to the bundled data/sample_images/<file>."""
    base = os.path.basename(row["image"])
    for d in ("images", "sample_images"):
        p = os.path.join(data_dir, d, base)
        if os.path.exists(p):
            return p
    return None


def count_images(data_dir=DATA):
    d = os.path.join(data_dir, "images")
    return sum(1 for n in os.listdir(d) if n.endswith(".png")) if os.path.isdir(d) else 0


def prf(y, p, k):
    """Accuracy and macro precision / recall / F1, in percent."""
    n = max(len(y), 1)
    out = {"acc": 100 * sum(a == b for a, b in zip(y, p)) / n}
    P = R = F = 0.0
    for c in range(k):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        P, R, F = P + pr, R + rc, F + (2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    out.update(precision=100 * P / k, recall=100 * R / k, f1=100 * F / k)
    return out


# ------------------------------------------------------------------ the model
class Analyzer:
    """Loads the trained cross-modal model once and answers (text, image) -> aspect + sentiment."""

    def __init__(self, ckpt_dir=CKPT):
        import torch
        from .models import build

        with open(os.path.join(ckpt_dir, "vocab.json"), encoding="utf-8") as f:
            self.itos = json.load(f)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.model = build("cross", len(self.itos))
        self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, "cross_seed0.pt"), map_location="cpu"))
        self.model.eval()

    # -- input preparation
    def _tokens(self, text):
        from .data import MAX_LEN, PAD, UNK, tokenize

        toks = tokenize(text)[:MAX_LEN]
        ids = [self.stoi.get(t, UNK) for t in toks]
        return toks, ids + [PAD] * (MAX_LEN - len(ids))

    @staticmethod
    def _pixels(image):
        import numpy as np
        from PIL import Image

        im = image.convert("RGB").resize((64, 64), Image.LANCZOS)
        return np.asarray(im).copy()

    # -- single prediction
    def predict(self, text, image):
        """image: PIL.Image or a file path.  Returns {'both','text','image'} -> aspect/sentiment + probabilities."""
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image

        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        toks, ids = self._tokens(text)
        tok = torch.tensor([ids])
        x = torch.from_numpy(self._pixels(image)).permute(2, 0, 1).float().unsqueeze(0).div(255).sub(.5).div(.5)
        out = {}
        with torch.no_grad():
            for name, drop in MODES:
                pa, ps = self.model(tok, x, drop=drop)
                a, s = F.softmax(pa, -1)[0].tolist(), F.softmax(ps, -1)[0].tolist()
                ia, is_ = int(np.argmax(a)), int(np.argmax(s))
                out[name] = {"aspect": ASPECTS[ia], "aspect_conf": a[ia], "aspect_probs": dict(zip(ASPECTS, a)),
                             "sentiment": SENTIMENTS[is_], "sentiment_conf": s[is_],
                             "sentiment_probs": dict(zip(SENTIMENTS, s))}
        out["tokens"] = toks
        out["unknown_tokens"] = [t for t in toks if t not in self.stoi]
        return out

    # -- evidence: which parts of the photo / which words drove the decision (occlusion, computed live)
    def explain(self, text, image):
        """Hide each photo cell (4x4) and each word in turn and measure how much the winning answer's probability drops."""
        import numpy as np
        import torch
        from PIL import Image

        from .data import UNK

        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        toks, ids = self._tokens(text)
        px = self._pixels(image)
        norm = lambda a: torch.from_numpy(np.asarray(a).copy()).permute(0, 3, 1, 2).float().div(255).sub(.5).div(.5)  # noqa: E731
        x0, tok0 = norm(px[None]), torch.tensor([ids])
        with torch.no_grad():
            pa, ps = self.model(tok0, x0)
            pa, ps = pa.softmax(-1)[0], ps.softmax(-1)[0]
            ca, cs = int(pa.argmax()), int(ps.argmax())
            cells = []
            for r in range(4):
                for c in range(4):
                    q = px.copy()
                    q[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = 128
                    cells.append(q)
            oa, os_ = self.model(tok0.repeat(16, 1), norm(np.stack(cells)))
            oa, os_ = oa.softmax(-1)[:, ca], os_.softmax(-1)[:, cs]
            words = []
            if toks:
                T = tok0.repeat(len(toks), 1)
                for j in range(len(toks)):
                    T[j, j] = UNK
                wa, ws = self.model(T, x0.repeat(len(toks), 1, 1, 1))
                words = [{"w": t, "aspect": float(pa[ca] - wa.softmax(-1)[j, ca]), "sentiment": float(ps[cs] - ws.softmax(-1)[j, cs])}
                         for j, t in enumerate(toks)]
        grid = lambda v, base: [[float(base - v[r * 4 + c]) for c in range(4)] for r in range(4)]  # noqa: E731
        return {"grid": 4, "aspect_drop": grid(oa, pa[ca]), "sentiment_drop": grid(os_, ps[cs]), "words": words,
                "model": "cross-modal network trained on the generated dataset",
                "params": int(sum(p.numel() for p in self.model.parameters()))}

    # -- many predictions at once (used by evaluate)
    def predict_labels(self, rows, data_dir=DATA, batch=128):
        import numpy as np
        import torch
        from PIL import Image

        toks, pix = [], []
        for r in rows:
            toks.append(self._tokens(r["text"])[1])
            pix.append(self._pixels(Image.open(image_path(r, data_dir))))
        tok = torch.tensor(toks)
        x = torch.from_numpy(np.stack(pix)).permute(0, 3, 1, 2).float().div(255).sub(.5).div(.5)
        res = {name: ([], []) for name, _ in MODES}
        with torch.no_grad():
            for i in range(0, len(rows), batch):
                for name, drop in MODES:
                    pa, ps = self.model(tok[i:i + batch], x[i:i + batch], drop=drop)
                    res[name][0].extend(pa.argmax(-1).tolist())
                    res[name][1].extend(ps.argmax(-1).tolist())
        return res


def evaluate(analyzer, rows, data_dir=DATA):
    """Accuracy / precision / recall / F1 (aspect and sentiment) + joint accuracy for each input mode."""
    rows = [r for r in rows if image_path(r, data_dir)]
    ya = [ASPECTS.index(r["aspect"]) for r in rows]
    ys = [SENTIMENTS.index(r["sentiment"]) for r in rows]
    pred = analyzer.predict_labels(rows, data_dir)
    out = {}
    for name, (pa, ps) in pred.items():
        joint = 100 * sum(a == b and c == d for a, b, c, d in zip(ya, pa, ys, ps)) / max(len(rows), 1)
        out[name] = {"aspect": prf(ya, pa, len(ASPECTS)), "sentiment": prf(ys, ps, len(SENTIMENTS)), "joint": joint}
    return {"n": len(rows), "result": out}
