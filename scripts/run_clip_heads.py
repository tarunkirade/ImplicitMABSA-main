"""Train small heads on FROZEN CLIP features (text-only, image-only, late, concat, gated fusion) - 3 seeds.

Why: the from-scratch text encoder cannot cope with unseen wording, and the MobileNet/spaCy features used for the
real tweets are weak.  CLIP features are far richer; only the small heads are trained here, on OUR data.

    python scripts/extract_clip_features.py --twitter-dir <TwitterDataMABSA>      # once
    python scripts/run_clip_heads.py                                              # -> results/clip_heads.json

Output keys follow results/summary.json (generated data) and results/real/real_results.json (real tweets).
"""
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from implicitmabsa.api import load_rows, prf  # noqa: E402
from implicitmabsa.lexicon import ASPECTS, SENTIMENTS  # noqa: E402

FEAT = os.path.join(HERE, "..", "features")
KINDS = ["text", "image", "late", "concat", "fusion"]


def mlp(i, h, o, p=0.3):
    return nn.Sequential(nn.Dropout(p), nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class Head(nn.Module):
    """kind: text | image | concat | fusion.  n_asp = 0 -> sentiment only (real tweets)."""

    def __init__(self, kind, n_asp, n_sent=3, d=256, mod_drop=0.15):
        super().__init__()
        self.kind, self.n_asp, self.mod_drop = kind, n_asp, mod_drop
        self.ln_t, self.ln_i = nn.LayerNorm(512), nn.LayerNorm(512)
        if kind == "concat":
            self.enc = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, d), nn.GELU())
        elif kind == "fusion":
            self.pt, self.pi = nn.Linear(512, d), nn.Linear(512, d)
            self.gate = nn.Linear(2 * d, d)
            self.enc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, d), nn.GELU())
        else:
            self.enc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, d), nn.GELU())
        self.asp = mlp(d, 128, n_asp) if n_asp else None
        self.sent = mlp(d + n_asp, 128, n_sent)

    def forward(self, t, i):
        t, i = self.ln_t(t), self.ln_i(i)
        if self.training and self.kind in ("concat", "fusion"):      # modality dropout
            r = torch.rand(t.shape[0], 1)
            t, i = t * (r >= self.mod_drop / 2), i * ((r < self.mod_drop / 2) | (r >= self.mod_drop))
        if self.kind == "text":
            h = self.enc(t)
        elif self.kind == "image":
            h = self.enc(i)
        elif self.kind == "concat":
            h = self.enc(torch.cat([t, i], -1))
        else:
            a, b = self.pt(t), self.pi(i)
            g = torch.sigmoid(self.gate(torch.cat([a, b], -1)))
            h = self.enc(torch.cat([g * a + (1 - g) * b, a * b], -1))   # gated mix + interaction term
        if self.asp is None:
            return None, self.sent(h)
        la = self.asp(h)
        return la, self.sent(torch.cat([h, la.softmax(-1)], -1))


def fit(kind, tr, va, n_asp, seed, epochs=60, bs=128):
    torch.manual_seed(seed)
    m = Head(kind, n_asp)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    n, best, state = len(tr["y_s"]), -1, None
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for k in range(0, n, bs):
            j = perm[k:k + bs]
            la, ls = m(tr["t"][j], tr["i"][j])
            loss = F.cross_entropy(ls, tr["y_s"][j], label_smoothing=0.1)
            if n_asp:
                loss = loss + F.cross_entropy(la, tr["y_a"][j], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sc = score(predict(m, va), va, n_asp)
        obj = (sc["aspect"]["f1"] + sc["sentiment"]["f1"]) if n_asp else sc["sentiment"]["f1"]
        if obj > best:
            best, state = obj, copy.deepcopy(m.state_dict())
    m.load_state_dict(state)
    return m


@torch.no_grad()
def predict(m, d):
    m.eval()
    la, ls = m(d["t"], d["i"])
    return (la.softmax(-1) if la is not None else None), ls.softmax(-1)


def score(p, d, n_asp):
    pa, ps = p
    out = {"sentiment": prf(d["y_s"].tolist(), ps.argmax(1).tolist(), 3)}
    if n_asp:
        out["aspect"] = prf(d["y_a"].tolist(), pa.argmax(1).tolist(), n_asp)
        out["joint"] = 100 * float(((pa.argmax(1) == d["y_a"]) & (ps.argmax(1) == d["y_s"])).float().mean())
    return out


def mean_std(runs, getter):
    v = [getter(r) for r in runs]
    return [float(np.mean(v)), float(np.std(v))]


def run_kinds(tr, va, tests, n_asp, seeds):
    """returns {test_name: {kind: [per-seed score dicts]}}"""
    res = {t: {k: [] for k in KINDS} for t in tests}
    for seed in seeds:
        probs = {t: {} for t in tests}
        for kind in ["text", "image", "concat", "fusion"]:
            m = fit(kind, tr, va, n_asp, seed)
            for t, d in tests.items():
                probs[t][kind] = predict(m, d)
                res[t][kind].append(score(probs[t][kind], d, n_asp))
        for t, d in tests.items():                                   # late fusion = average the two heads' probabilities
            a, b = probs[t]["text"], probs[t]["image"]
            late = ((a[0] + b[0]) / 2 if n_asp else None, (a[1] + b[1]) / 2)
            res[t]["late"].append(score(late, d, n_asp))
        print("  seed", seed, {k: round(res[next(iter(tests))][k][-1]["sentiment"]["f1"], 1) for k in KINDS}, flush=True)
    return res


def synthetic(seeds):
    z = np.load(os.path.join(FEAT, "clip_synth.npz"))
    rows = {r["id"]: r for r in load_rows()}
    ids = [str(x) for x in z["ids"]]
    T, I = torch.tensor(z["txt"].astype(np.float32)), torch.tensor(z["img"].astype(np.float32))
    sp = np.array([rows[i]["split"] for i in ids])
    ya = torch.tensor([ASPECTS.index(rows[i]["aspect"]) for i in ids])
    ys = torch.tensor([SENTIMENTS.index(rows[i]["sentiment"]) for i in ids])

    def sub(name):
        k = torch.tensor(np.where(sp == name)[0])
        return {"t": T[k], "i": I[k], "y_a": ya[k], "y_s": ys[k]}

    res = run_kinds(sub("train"), sub("val"), {"test_iid": sub("test_iid"), "test_ood": sub("test_ood")}, 8, seeds)
    out = {}
    for t, per in res.items():
        out[t] = {}
        for k, runs in per.items():
            o = {}
            for task, key in (("aspect", "aspect"), ("sentiment", "sentiment")):
                for met in ("acc", "precision", "recall", "f1"):
                    o[f"{key}_{met}"] = mean_std(runs, lambda r, task=task, met=met: r[task][met])
            o["joint_acc"] = mean_std(runs, lambda r: r["joint"])
            out[t][k] = o
    return out


def twitter(year, variant, seeds):
    z = np.load(os.path.join(FEAT, f"clip_tw{year}.npz"))
    img = torch.tensor(z["img"].astype(np.float32))

    def get(sp):
        idx = torch.tensor(z[f"{sp}_img_idx"]).long()
        i = img[idx.clamp_min(0)] * (idx >= 0).float().unsqueeze(1)
        return {"t": torch.tensor(z[f"{sp}_txt_{variant}"].astype(np.float32)), "i": i, "y_s": torch.tensor(z[f"{sp}_label"]).long()}

    tr, va, te = get("train"), get("val"), get("test")
    maj = int(np.bincount(tr["y_s"].numpy()).argmax())
    out = {"majority": prf(te["y_s"].tolist(), [maj] * len(te["y_s"]), 3)}
    res = run_kinds(tr, va, {"test": te}, 0, seeds)["test"]
    for k, runs in res.items():
        out[k] = {met: mean_std(runs, lambda r, met=met: r["sentiment"][met]) for met in ("acc", "precision", "recall", "f1")}
    return out


def main():
    torch.set_num_threads(2)
    seeds = [0, 1, 2]
    res = {}
    if os.path.exists(os.path.join(FEAT, "clip_synth.npz")):
        print("generated dataset", flush=True)
        res["synthetic"] = synthetic(seeds)
        for t in ("test_iid", "test_ood"):
            f = res["synthetic"][t]["fusion"]
            print(" ", t, "fusion head: aspect F1 %.1f  sentiment F1 %.1f" % (f["aspect_f1"][0], f["sentiment_f1"][0]), flush=True)
    res["twitter"] = {}
    for year in ("2015", "2017"):
        if os.path.exists(os.path.join(FEAT, f"clip_tw{year}.npz")):
            for variant in ("explicit", "masked"):
                print("Twitter", year, variant, flush=True)
                res["twitter"][f"{year}_{variant}"] = twitter(year, variant, seeds)
    with open(os.path.join(HERE, "..", "results", "clip_heads.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved results/clip_heads.json")


if __name__ == "__main__":
    main()
