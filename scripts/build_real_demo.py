"""Build the small REAL-data sample that the UI shows (Twitter-2015 / Twitter-2017 test tweets).

What it does
  1. trains one model of each kind (text-only, image-only, concat fusion, cross-modal) on the real training split,
     using the frozen pretrained features made by scripts/extract_features.py  (sentiment: negative/neutral/positive)
  2. picks N random test tweets per year, saves a small thumbnail of each image
  3. stores every model's predicted probabilities next to the true label  ->  data/real/samples.json

So the UI can show real tweets with what each model said, with no big download and no heavy libraries.

    python scripts/extract_features.py --twitter-dir <TwitterDataMABSA> --year 2015 --weights weights/mobilenetv3.pth
    python scripts/build_real_demo.py  --twitter-dir <TwitterDataMABSA> --years 2015 2017 --n 400
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from run_real import fit, load, probs  # noqa: E402

LABELS = ["negative", "neutral", "positive"]


def read_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 5:
                rows.append({"label": int(p[1]), "image": p[2], "text": p[3], "target": p[4]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twitter-dir", required=True)
    ap.add_argument("--years", nargs="+", default=["2015", "2017"])
    ap.add_argument("--n", type=int, default=400, help="tweets kept per year")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "real"))
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(os.path.join(a.out, "thumbs"), exist_ok=True)
    samples = []
    for year in a.years:
        print(f"== Twitter-{year}: training", flush=True)
        D, emb = load(year, "explicit")
        rows = read_tsv(os.path.join(a.twitter_dir, f"twitter{year}", "test.tsv"))
        assert len(rows) == len(D["test"][2]), "TSV and features disagree"
        P = {}
        for name in ["text", "image", "concat", "cross"]:
            m = fit(name, D, emb, seed=0, epochs=a.epochs)
            P[name] = probs(m, D["test"])
            print("  ", name, "test acc %.1f" % (100 * (P[name].argmax(1) == D["test"][2].numpy()).mean()), flush=True)
        P["late"] = (P["text"] + P["image"]) / 2
        idx = sorted(random.Random(1).sample(range(len(rows)), min(a.n, len(rows))))
        for i in idx:
            r = rows[i]
            src = os.path.join(a.twitter_dir, f"twitter{year}_images", r["image"])
            sid = f"t{year}_{i:04d}"
            try:
                im = Image.open(src).convert("RGB")
                im.thumbnail((160, 160))
                im.save(os.path.join(a.out, "thumbs", sid + ".jpg"), quality=72)
            except Exception:
                continue
            samples.append({
                "id": sid, "year": year, "text": r["text"], "target": r["target"], "label": LABELS[r["label"]],
                "probs": {k: [round(float(x), 4) for x in v[i]] for k, v in P.items()},
            })
    with open(os.path.join(a.out, "samples.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "samples": samples}, f, ensure_ascii=False)
    print("saved", len(samples), "real tweets ->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
