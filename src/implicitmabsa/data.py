"""Data loading: tokeniser, vocabulary, image tensors, batching."""
import csv
import os
import re
from collections import Counter

import numpy as np
import torch
from PIL import Image

from .lexicon import ASPECTS, SENTIMENTS

PAD, UNK = 0, 1
MAX_LEN = 24
_TOK = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return _TOK.findall(text.lower())


class Vocab:
    def __init__(self, texts, min_freq=1):
        c = Counter(t for s in texts for t in tokenize(s))
        self.itos = ["<pad>", "<unk>"] + sorted(w for w, f in c.items() if f >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text):
        ids = [self.stoi.get(t, UNK) for t in tokenize(text)][:MAX_LEN]
        return ids + [PAD] * (MAX_LEN - len(ids))

    def __len__(self):
        return len(self.itos)


class Split:
    """Holds one split as tensors."""

    def __init__(self, rows, vocab, root):
        self.rows = rows
        self.tokens = torch.tensor([vocab.encode(r["text"]) for r in rows], dtype=torch.long)
        self.images = torch.from_numpy(np.stack([
            np.asarray(Image.open(os.path.join(root, r["image"])).convert("RGB")) for r in rows
        ])).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,64,64
        self.aspect = torch.tensor([ASPECTS.index(r["aspect"]) for r in rows])
        self.sent = torch.tensor([SENTIMENTS.index(r["sentiment"]) for r in rows])

    def __len__(self):
        return len(self.rows)

    def batches(self, bs, shuffle=False, augment=False, gen=None):
        n = len(self)
        idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            img = self.images[j].float().div_(255).sub_(0.5).div_(0.5)
            if augment:  # random +-3px translation (same shift per batch: cheap regulariser)
                dx, dy = torch.randint(-3, 4, (2,), generator=gen).tolist()
                img = torch.roll(img, shifts=(dy, dx), dims=(2, 3))
            yield self.tokens[j], img, self.aspect[j], self.sent[j], j


def load_dataset(root, min_freq=1):
    with open(os.path.join(root, "metadata.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = {k: [r for r in rows if r["split"] == k] for k in ["train", "val", "test_iid", "test_ood"]}
    by = {k: v for k, v in by.items() if v}          # real-data corpora may have no OOD split
    vocab = Vocab([r["text"] for r in by["train"]], min_freq=min_freq)
    return {k: Split(v, vocab, root) for k, v in by.items()}, vocab
