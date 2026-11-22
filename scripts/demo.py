"""Interactive demo: predict the implicit aspect and sentiment for any (text, image) pair.

    python scripts/demo.py --text "Dead by lunch again" --image data/images/test_iid_00000.png
    python scripts/demo.py --text "Not what I expected!" --image my_photo.jpg      # image supplies the aspect
    python scripts/demo.py --text "Not what I expected!" --image my_photo.jpg --compare   # + text-only / image-only
Requires checkpoints produced by scripts/run_experiments.py.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from implicitmabsa.data import MAX_LEN, PAD, UNK, tokenize  # noqa: E402
from implicitmabsa.lexicon import ASPECTS, SENTIMENTS  # noqa: E402
from implicitmabsa.models import build  # noqa: E402


def load(name, ckpt, vocab):
    m = build(name, len(vocab))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--compare", action="store_true", help="also show text-only / image-only inference of the same model")
    a = ap.parse_args()
    itos = json.load(open(os.path.join(a.ckpt, "vocab.json"), encoding="utf-8"))
    stoi = {w: i for i, w in enumerate(itos)}
    ids = [stoi.get(t, UNK) for t in tokenize(a.text)][:MAX_LEN]
    tok = torch.tensor([ids + [PAD] * (MAX_LEN - len(ids))])
    img = Image.open(a.image).convert("RGB").resize((64, 64), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().unsqueeze(0).div(255).sub(.5).div(.5)
    model = load("cross", os.path.join(a.ckpt, "cross_seed0.pt"), itos)

    def show(title, drop=None):
        with torch.no_grad():
            pa, ps = model(tok, x, drop=drop)
        pa, ps = F.softmax(pa, -1)[0], F.softmax(ps, -1)[0]
        print(f"{title:<22} aspect = {ASPECTS[pa.argmax()]:<12} ({pa.max():.2f})   "
              f"sentiment = {SENTIMENTS[ps.argmax()]:<9} ({ps.max():.2f})")

    print(f'text : "{a.text}"\nimage: {a.image}')
    show("multimodal (text+image)")
    if a.compare:
        show("text only  (image off)", drop="image")
        show("image only (text off)", drop="text")


if __name__ == "__main__":
    main()
