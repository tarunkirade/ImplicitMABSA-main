"""Frozen CLIP features (512-d image + 512-d text) for the generated dataset and the real Twitter corpora.
Used by scripts/run_clip_heads.py, which trains small heads on top.

    python scripts/extract_clip_features.py --twitter-dir <TwitterDataMABSA>
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from implicitmabsa.api import image_path, load_rows  # noqa: E402
from implicitmabsa.clip_engine import ClipAnalyzer  # noqa: E402


def read_tsv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            x = line.rstrip("\n").split("\t")
            if len(x) >= 5:
                rows.append({"label": int(x[1]), "image": x[2], "text": x[3], "target": x[4]})
    return rows


def imgs(an, paths, tag):
    out = []
    for i in range(0, len(paths), 256):
        out.append(an.img([Image.open(p) for p in paths[i:i + 256]]).numpy().astype(np.float16))
        print(f"  {tag} images {min(i + 256, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(out)


def texts(an, strs, tag):
    out = []
    for i in range(0, len(strs), 512):
        out.append(an.txt(strs[i:i + 512]).numpy().astype(np.float16))
        print(f"  {tag} texts {min(i + 512, len(strs))}/{len(strs)}", flush=True)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twitter-dir", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "features"))
    a = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(a.out, exist_ok=True)
    an = ClipAnalyzer()
    rows = [r for r in load_rows() if image_path(r)]
    print("generated dataset:", len(rows), flush=True)
    np.savez_compressed(os.path.join(a.out, "clip_synth.npz"), ids=np.array([r["id"] for r in rows]),
                        img=imgs(an, [image_path(r) for r in rows], "synth"), txt=texts(an, [r["text"] for r in rows], "synth"))
    if a.twitter_dir:
        for year in ("2015", "2017"):
            d, idir = os.path.join(a.twitter_dir, f"twitter{year}"), os.path.join(a.twitter_dir, f"twitter{year}_images")
            data = {sp: read_tsv(os.path.join(d, f)) for sp, f in (("train", "train.tsv"), ("val", "dev.tsv"), ("test", "test.tsv"))}
            names = sorted({r["image"] for v in data.values() for r in v})
            ok = [n for n in names if os.path.exists(os.path.join(idir, n))]
            fi = imgs(an, [os.path.join(idir, n) for n in ok], f"tw{year}")
            save = {"img": fi, "img_names": np.array(ok)}
            for sp, v in data.items():
                save[f"{sp}_label"] = np.array([r["label"] for r in v])
                save[f"{sp}_img_idx"] = np.array([ok.index(r["image"]) if r["image"] in ok else -1 for r in v])
                save[f"{sp}_txt_explicit"] = texts(an, [r["text"].replace("$T$", r["target"]) for r in v], f"tw{year}-{sp}-explicit")
                save[f"{sp}_txt_masked"] = texts(an, [r["text"].replace("$T$", "[MASK]") for r in v], f"tw{year}-{sp}-masked")
            np.savez_compressed(os.path.join(a.out, f"clip_tw{year}.npz"), **save)
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
