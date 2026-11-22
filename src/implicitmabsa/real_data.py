"""Adapters for REAL multimodal ABSA corpora (to be used where the images can be downloaded).

The synthetic benchmark in this repository exists because the public corpora are hosted on Google Drive /
Baidu and could not be fetched in the environment used to build the project.  These adapters convert the
real corpora into the same ``metadata.csv`` + ``images/`` layout that ``load_dataset`` consumes, so the
whole pipeline (training, evaluation, analysis) runs unchanged.

Supported
---------
* Twitter-2015 / Twitter-2017  (Zhang et al. 2018, Lu et al. 2018; TSV files from
  https://github.com/jefferyYu/TomBERT plus the image folders from that repository's README link).
  Each line: index, sentiment(0 neg/1 neu/2 pos), image_id, text with "$T$" placeholder, aspect term.
  IMPLICIT VARIANT: the aspect *term* is removed from the sentence (replaced by "[MASK-ASPECT]") so the
  model must infer the target from the remaining text and the image; the sentiment label is unchanged.
  CAVEAT: Twitter targets are named ENTITIES (e.g. "chicago", "# blackhawks"), not aspect categories, so
  the "aspect" column produced here is only a proxy.  Use this adapter for the *sentiment* task with the
  target masked (compare text / image / fusion on real data); the aspect-classification numbers are NOT
  meaningful on Twitter.  Corpora with true aspect categories (MASAD, Multi-ZOL, MACSA) need an analogous
  ~30-line adapter that maps their category field to the ``aspect`` column.

Usage
-----
    python -m implicitmabsa.real_data --tsv-dir absa_data/twitter2015 --img-dir IJCAI2019_data/twitter2015_images \
           --out data_twitter2015 --top-k 8
Then train with  ``python scripts/run_experiments.py --data data_twitter2015``  after editing
``lexicon.ASPECTS`` (or pass the class list) - see README section "Using real data".
"""
import argparse
import csv
import os
import shutil
from collections import Counter

from PIL import Image

SENT = {"0": "negative", "1": "neutral", "2": "positive"}


def read_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            _, s, img, text, aspect = p[:5]
            rows.append((SENT[s], img, text, aspect.strip().lower()))
    return rows


def convert(tsv_dir, img_dir, out, top_k=8, mask="[MASK-ASPECT]"):
    splits = {"train": "train.tsv", "val": "dev.tsv", "test_iid": "test.tsv"}
    data = {k: read_tsv(os.path.join(tsv_dir, v)) for k, v in splits.items()}
    freq = Counter(a for r in data["train"] for _, _, _, a in [r])
    keep = [a for a, _ in freq.most_common(top_k)]
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    out_rows, missing = [], 0
    for split, rows in data.items():
        for i, (sent, img, text, asp) in enumerate(rows):
            if asp not in keep:
                continue
            src = os.path.join(img_dir, img)
            if not os.path.exists(src):
                missing += 1
                continue
            fn = f"{split}_{i:05d}.png"
            Image.open(src).convert("RGB").resize((64, 64), Image.LANCZOS).save(os.path.join(out, "images", fn))
            out_rows.append(dict(id=f"{split}_{i:05d}", split=split, text=text.replace("$T$", mask),
                                 image=f"images/{fn}", aspect=asp, sentiment=sent,
                                 text_has_aspect=0, text_has_sent=1, img_has_aspect=0, img_has_sent=1,
                                 aspect_src="B", sent_src="B", conflict=0))
    with open(os.path.join(out, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows, classes={keep}, missing images={missing}")
    return keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    a = ap.parse_args()
    convert(a.tsv_dir, a.img_dir, a.out, a.top_k)

