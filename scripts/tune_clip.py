"""Tune the few CLIP-engine settings on the VALIDATION split, then report once on the two test splits.

    python scripts/tune_clip.py            -> results/clip_eval.json
"""
import itertools
import json
import os
import sys

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from implicitmabsa.api import image_path, load_rows, prf  # noqa: E402
from implicitmabsa.clip_engine import ClipAnalyzer  # noqa: E402
from implicitmabsa.lexicon import ASPECTS, SENTIMENTS  # noqa: E402


def feats(an, split):
    rows = [r for r in load_rows(split=split) if image_path(r)]
    fi = an.img([Image.open(image_path(r)) for r in rows])
    ft = an.txt([r["text"] for r in rows])
    return rows, fi, ft


def score(an, rows, fi, ft):
    texts = [r["text"] for r in rows]
    ya = [ASPECTS.index(r["aspect"]) for r in rows]
    ys = [SENTIMENTS.index(r["sentiment"]) for r in rows]
    out = {}
    for name, (a, s) in an.analyse_features(fi, ft, texts).items():
        pa, ps = a.argmax(1).tolist(), s.argmax(1).tolist()
        out[name] = {"aspect": prf(ya, pa, 8), "sentiment": prf(ys, ps, 3),
                     "joint": 100 * sum(x == y and u == v for x, y, u, v in zip(ya, pa, ys, ps)) / len(rows)}
    return out


def main():
    torch.set_num_threads(2)
    an = ClipAnalyzer(topics=ASPECTS)           # the 8-topic benchmark setting
    val = feats(an, "val")
    best, grid = None, itertools.product([0.3, 0.6, 1.0, 1.5], [2.0, 3.0, 5.0], [0.3, 0.8, 1.5], [0.8, 1.0])
    for w, k, nb, t in grid:
        an.p.update(w_text=w, k_sent=k, neutral_bias=nb, img_t=t)
        r = score(an, *val)["both"]
        obj = r["aspect"]["f1"] + r["sentiment"]["f1"]
        if best is None or obj > best[0]:
            best = (obj, dict(w_text=w, k_sent=k, neutral_bias=nb, img_t=t), r)
    print("best on validation:", best[1], "aspect F1 %.1f sentiment F1 %.1f" % (best[2]["aspect"]["f1"], best[2]["sentiment"]["f1"]), flush=True)
    an.p.update(best[1])
    res = {"chosen_on_validation": best[1], "validation": score(an, *val)}
    for split in ("test_iid", "test_ood"):
        res[split] = score(an, *feats(an, split))
        b = res[split]["both"]
        print(split, "both: aspect F1 %.1f  sentiment F1 %.1f | text-only %.1f/%.1f | image-only %.1f/%.1f" % (
            b["aspect"]["f1"], b["sentiment"]["f1"], res[split]["text"]["aspect"]["f1"], res[split]["text"]["sentiment"]["f1"],
            res[split]["image"]["aspect"]["f1"], res[split]["image"]["sentiment"]["f1"]), flush=True)
    json.dump(res, open(os.path.join(HERE, "..", "results", "clip_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
