"""Training / evaluation utilities."""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .models import build


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def predict(model, split, bs=256, drop=None):
    model.eval()
    A, S = [], []
    for tok, img, _, _, _ in split.batches(bs):
        a, s = model(tok, img, drop=drop)
        A.append(F.softmax(a, -1))
        S.append(F.softmax(s, -1))
    return torch.cat(A).numpy(), torch.cat(S).numpy()


def metrics(y, prob):
    p = prob.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(acc=accuracy_score(y, p), precision=pr, recall=rc, f1=f1)


def evaluate_probs(split, pa, ps):
    ya, ys = split.aspect.numpy(), split.sent.numpy()
    out = {"aspect": metrics(ya, pa), "sentiment": metrics(ys, ps)}
    out["joint_acc"] = float(((pa.argmax(1) == ya) & (ps.argmax(1) == ys)).mean())
    return out


def fit(name, data, vocab_size, seed=0, epochs=15, bs=64, lr=2e-3, log=print):
    set_seed(seed)
    model = build(name, vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(data["train"]) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, t0 = -1, None, time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for tok, img, ya, ys, _ in data["train"].batches(bs, shuffle=True, augment=True, gen=gen):
            a, s = model(tok, img)
            loss = F.cross_entropy(a, ya, label_smoothing=0.05) + F.cross_entropy(s, ys, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(ya)
        pa, ps = predict(model, data["val"])
        m = evaluate_probs(data["val"], pa, ps)
        score = (m["aspect"]["f1"] + m["sentiment"]["f1"]) / 2
        if score > best:
            best, best_state = score, copy.deepcopy(model.state_dict())
        log(f"[{name} s{seed}] ep{ep + 1:02d} loss {tot / len(data['train']):.3f} "
            f"val aspF1 {m['aspect']['f1']:.3f} sentF1 {m['sentiment']['f1']:.3f} ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    return model, best
