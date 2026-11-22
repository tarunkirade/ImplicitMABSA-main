"""Extract FROZEN pretrained features for the real Twitter-2015 / Twitter-2017 corpora.

  text : spaCy en_core_web_md word vectors (300-d, GloVe-style)   -> token ids + embedding matrix
  image: timm MobileNetV3-Large (ImageNet weights) -> 3x3 grid of 960-d region features

Two text variants per example
  explicit : "$T$" replaced by the target term            (standard target-oriented setting)
  masked   : "$T$" replaced by a MASK token               (IMPLICIT target: the model must infer what the
                                                           sentiment is about from context + image)
Data: https://github.com/CopotronicRifat/TwitterDataMABSA  (TSV files + image folders; from Yu & Jiang, IJCAI'19).
Usage: python scripts/extract_features.py --twitter-dir TwitterDataMABSA --year 2015 --weights weights/mobilenetv3.pth
"""
import argparse
import csv
import os

import numpy as np
import torch
import timm
from PIL import Image

MAXLEN = 40
SENT = {"0": 0, "1": 1, "2": 2}  # negative, neutral, positive


def read_tsv(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            x = line.rstrip("\n").split("\t")
            if len(x) >= 5:
                rows.append(dict(label=SENT[x[1]], image=x[2], text=x[3], target=x[4]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twitter-dir", required=True)
    ap.add_argument("--year", default="2015")
    ap.add_argument("--weights", default="weights/mobilenetv3.pth")
    ap.add_argument("--out", default="features")
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("en_core_web_md")
    torch.set_num_threads(2)
    d = os.path.join(a.twitter_dir, f"twitter{a.year}")
    imgdir = os.path.join(a.twitter_dir, f"twitter{a.year}_images")
    splits = {"train": "train.tsv", "val": "dev.tsv", "test": "test.tsv"}
    data = {k: read_tsv(os.path.join(d, v)) for k, v in splits.items()}

    # ---- text
    rng = np.random.RandomState(0)
    vocab = {"<pad>": 0, "<mask>": 1}
    vecs = [np.zeros(300, np.float32), rng.normal(0, .3, 300).astype(np.float32)]

    def ids(text):
        out = []
        for t in text.lower().split()[:MAXLEN]:
            if t not in vocab:
                vocab[t] = len(vocab)
                vecs.append(nlp.vocab[t].vector.astype(np.float32) if nlp.vocab[t].has_vector else
                            np.zeros(300, np.float32))
            out.append(vocab[t])
        return out + [0] * (MAXLEN - len(out))

    out = {}
    for sp, rows in data.items():
        e, m = [], []
        for r in rows:
            t = r["text"]
            e.append(ids(t.replace("$T$", r["target"])))
            m.append(ids(t.replace("$T$", "<mask>")))
        # NOTE: "<mask>" must map to index 1: lower()-split gives token "<mask>" which we set below
        out[sp] = dict(ids_explicit=np.array(e), ids_masked=np.array(m),
                       label=np.array([r["label"] for r in rows]), image=[r["image"] for r in rows])
    # ---- images (dedupe)
    all_imgs = sorted({r for sp in out.values() for r in sp["image"]})
    index = {n: i for i, n in enumerate(all_imgs)}
    model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=0, global_pool="")
    sd = torch.load(a.weights, map_location="cpu")
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("classifier")}, strict=False)
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats = np.zeros((len(all_imgs), 9, 960), np.float16)
    ok = np.ones(len(all_imgs), bool)
    bs = 32
    for i in range(0, len(all_imgs), bs):
        batch = []
        for n in all_imgs[i:i + bs]:
            try:
                im = Image.open(os.path.join(imgdir, n)).convert("RGB").resize((224, 224), Image.BILINEAR)
            except Exception:
                im = Image.new("RGB", (224, 224), (128, 128, 128))
                ok[index[n]] = False
            batch.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
        x = (torch.stack(batch).float() / 255 - mean) / std
        with torch.no_grad():
            f = model.forward_features(x)                       # B,960,7,7
            g = torch.nn.functional.adaptive_avg_pool2d(f, 3)   # B,960,3,3
        feats[i:i + len(batch)] = g.flatten(2).transpose(1, 2).numpy().astype(np.float16)
        if (i // bs) % 20 == 0:
            print(f"images {i}/{len(all_imgs)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    save = dict(emb=np.stack(vecs), img=feats, img_ok=ok)
    for sp, o in out.items():
        for k in ("ids_explicit", "ids_masked", "label"):
            save[f"{sp}_{k}"] = o[k]
        save[f"{sp}_img_idx"] = np.array([index[n] for n in o["image"]])
    np.savez_compressed(os.path.join(a.out, f"twitter{a.year}.npz"), **save)
    print("saved", {sp: len(o["label"]) for sp, o in out.items()}, "images", len(all_imgs), "missing", int((~ok).sum()),
          "vocab", len(vocab))


if __name__ == "__main__":
    main()
