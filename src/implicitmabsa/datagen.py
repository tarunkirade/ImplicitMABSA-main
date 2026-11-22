"""Generate the ImplicitMABSA-Synth dataset (text + image + implicit-aspect + sentiment labels).

Design goals
------------
* The ASPECT is never named in the text (see lexicon.EXPLICIT_TERMS and tests/test_data.py).
* Information is spread across modalities so that neither modality alone is sufficient:
    aspect source    in {text only, image only, both}
    sentiment source in {text only, image only, both}
  and every sample has the aspect AND the sentiment recoverable from at least one modality.
* Realistic imperfections: typos, emojis (15% misleading), openers/closers, background clutter,
  jitter, blur, 10% text-image sentiment conflicts (label follows the text) and 2% label noise.
* Two test sets: `test_iid` (phrasings seen in training) and `test_ood` (held-out phrasings,
  new openers/closers) to measure generalisation to unseen wording.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter

from . import lexicon as L
from .imagegen import render

SRC = ["T", "I", "B"]
ASPECT_SRC_W = [0.35, 0.30, 0.35]
SENT_SRC_W = [0.30, 0.30, 0.40]
CONFLICT_P = 0.10      # when both modalities carry sentiment, prob. that the image disagrees
LABEL_NOISE_P = 0.02
EMOJI_P, EMOJI_WRONG_P, TYPO_P = 0.35, 0.15, 0.04


def _typo(s, rng):
    if len(s) < 6:
        return s
    i = rng.randint(1, len(s) - 3)
    if rng.random() < .5:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]   # swap
    return s[:i] + s[i + 1:]                          # delete


def make_text(aspect, sent, has_a, has_s, ood, rng):
    lo, hi = (L.N_SEEN, 8) if ood else (0, L.N_SEEN)
    if has_a and has_s:
        core = L.PHRASES[aspect][sent][rng.randrange(lo, hi)]
    elif has_a:
        tl = L.PHRASES[aspect]["topic"]
        core = tl[rng.randrange(3, 4) if ood else rng.randrange(0, 3)]
    elif has_s:
        core = L.GENERIC_SENT[sent][rng.randrange(lo, hi)]
    else:
        core = L.FILLER[rng.randrange(lo, hi)]
    op = rng.choice(L.OPENERS_OOD if ood else L.OPENERS_SEEN)
    cl = rng.choice(L.CLOSERS_OOD if ood else L.CLOSERS_SEEN)
    text = op + (core[0].lower() + core[1:] if op and not core.startswith(("I", "Grandma")) else core) + cl
    if has_s and rng.random() < EMOJI_P:
        es = sent if rng.random() > EMOJI_WRONG_P else rng.choice([x for x in L.SENTIMENTS if x != sent])
        text += " " + rng.choice(L.EMOJI[es])
    if rng.random() < TYPO_P:
        text = _typo(text, rng)
    return text


def make_sample(idx, split, rng, out_img_dir, seed_base):
    aspect = rng.choice(L.ASPECTS)
    sent = rng.choice(L.SENTIMENTS)
    a_src = rng.choices(SRC, ASPECT_SRC_W)[0]
    s_src = rng.choices(SRC, SENT_SRC_W)[0]
    t_a, t_s = a_src in "TB", s_src in "TB"
    i_a, i_s = a_src in "IB", s_src in "IB"

    text = make_text(aspect, sent, t_a, t_s, split == "test_ood", rng)

    conflict = False
    img_sent = sent
    if t_s and i_s and rng.random() < CONFLICT_P:
        conflict = True
        img_sent = rng.choice([x for x in L.SENTIMENTS if x != sent])
    if i_a and i_s:
        mode, state = "icon", img_sent
    elif i_a:
        mode, state = "icon", "amb"
    elif i_s:
        mode, state = "face", img_sent
    else:
        mode, state = "none", "amb"
    img = render(aspect if mode == "icon" else None, state, mode, seed_base + idx)
    fname = f"{split}_{idx:05d}.png"
    img.save(os.path.join(out_img_dir, fname), optimize=True)

    label_sent = sent
    if rng.random() < LABEL_NOISE_P:
        label_sent = rng.choice([x for x in L.SENTIMENTS if x != sent])
    return dict(id=f"{split}_{idx:05d}", split=split, text=text, image=f"images/{fname}",
                aspect=aspect, sentiment=label_sent,
                text_has_aspect=int(t_a), text_has_sent=int(t_s),
                img_has_aspect=int(i_a), img_has_sent=int(i_s),
                aspect_src=a_src, sent_src=s_src, conflict=int(conflict))


def generate(out_dir, n_train=5600, n_val=800, n_test_iid=800, n_test_ood=800, seed=13):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    rows = []
    for k, (split, n) in enumerate([("train", n_train), ("val", n_val),
                                    ("test_iid", n_test_iid), ("test_ood", n_test_ood)]):
        rng = random.Random(seed * 1000 + k)
        for i in range(n):
            rows.append(make_sample(i, split, rng, img_dir, seed_base=seed * 10_000_000 + k * 100_000))
    with open(os.path.join(out_dir, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stats = summarize(rows)
    with open(os.path.join(out_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return rows, stats


def summarize(rows):
    out = {}
    for split in ["train", "val", "test_iid", "test_ood"]:
        r = [x for x in rows if x["split"] == split]
        out[split] = dict(
            n=len(r),
            aspect=dict(Counter(x["aspect"] for x in r)),
            sentiment=dict(Counter(x["sentiment"] for x in r)),
            aspect_source=dict(Counter(x["aspect_src"] for x in r)),
            sentiment_source=dict(Counter(x["sent_src"] for x in r)),
            conflicts=sum(x["conflict"] for x in r),
            avg_words=round(sum(len(x["text"].split()) for x in r) / max(len(r), 1), 2),
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    _, st = generate(a.out, seed=a.seed)
    print(json.dumps(st, indent=1))
