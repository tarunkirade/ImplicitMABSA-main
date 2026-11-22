"""Real-data experiment: Twitter-2015 / 2017, sentiment, explicit vs implicit(masked)-target text.
   python scripts/run_real.py --years 2015 2017 --seeds 0 1 2 3 4"""
import argparse, json, os, sys, copy
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from implicitmabsa.real_models import build

def metrics(y, p):
    P, R, F1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(acc=accuracy_score(y, p) * 100, precision=P * 100, recall=R * 100, f1=F1 * 100)

def load(year, variant):
    z = np.load(f"features/twitter{year}.npz")
    img = torch.from_numpy(z["img"].astype(np.float32))
    D = {}
    for sp in ["train", "val", "test"]:
        D[sp] = (torch.from_numpy(z[f"{sp}_ids_{variant}"]).long(), img[torch.from_numpy(z[f"{sp}_img_idx"]).long()],
                 torch.from_numpy(z[f"{sp}_label"]).long())
    return D, z["emb"]

@torch.no_grad()
def probs(m, d, bs=256):
    m.eval(); out = []
    for i in range(0, len(d[2]), bs):
        out.append(F.softmax(m(d[0][i:i+bs], d[1][i:i+bs]), -1))
    return torch.cat(out).numpy()

def fit(name, D, emb, seed, epochs=12, bs=64, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(name, emb)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
    n = len(D["train"][2]); best, state = -1, None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i+bs]
            loss = F.cross_entropy(m(D["train"][0][j], D["train"][1][j]), D["train"][2][j], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
        f1 = metrics(D["val"][2].numpy(), probs(m, D["val"]).argmax(1))["f1"]
        if f1 > best: best, state = f1, copy.deepcopy(m.state_dict())
    m.load_state_dict(state)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", default=["2015", "2017"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args(); torch.set_num_threads(2); os.makedirs(a.out, exist_ok=True)
    res = {}
    for year in a.years:
        for variant in ["explicit", "masked"]:
            D, emb = load(year, variant)
            yt = D["test"][2].numpy()
            maj = np.bincount(D["train"][2].numpy()).argmax()
            R = {"majority": metrics(yt, np.full_like(yt, maj))}
            per = {n: [] for n in ["text", "image", "late", "concat", "cross"]}
            for seed in a.seeds:
                P = {}
                for n in ["text", "image", "concat", "cross"]:
                    m = fit(n, D, emb, seed); P[n] = probs(m, D["test"]); per[n].append(metrics(yt, P[n].argmax(1)))
                per["late"].append(metrics(yt, ((P["text"] + P["image"]) / 2).argmax(1)))
                print(year, variant, seed, {n: round(per[n][-1]["acc"], 1) for n in per}, flush=True)
            for n, L in per.items():
                R[n] = {k: [float(np.mean([x[k] for x in L])), float(np.std([x[k] for x in L]))] for k in L[0]}
            res[f"{year}_{variant}"] = R
            json.dump(res, open(f"{a.out}/real_results.json", "w"), indent=1)
if __name__ == "__main__":
    main()
