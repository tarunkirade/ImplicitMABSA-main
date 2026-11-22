"""Train every model for several seeds and save metrics + predicted probabilities.

    python scripts/run_experiments.py --data data --out results --seeds 0 1 2 --epochs 15
Resumable: finished (model, seed) pairs are skipped.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from implicitmabsa.data import load_dataset  # noqa: E402
from implicitmabsa.train import evaluate_probs, fit, predict  # noqa: E402

MODELS = ["text", "image", "concat", "cross", "cross_noasp", "cross_nomd"]
FUSION = {"concat", "cross", "cross_noasp", "cross_nomd"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("--sentiment-only", action="store_true", help="real data without aspect labels (aspect head unused)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    data, vocab = load_dataset(a.data, a.min_freq)
    if not a.sentiment_only:
        json.dump(vocab.itos, open("checkpoints/vocab.json", "w"), ensure_ascii=False)

    log_f = open(os.path.join(a.out, "train_log.txt"), "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    for seed in a.seeds:
        for name in a.models:
            tag = f"{name}_s{seed}"
            path = os.path.join(a.out, "runs", tag + ".json")
            if os.path.exists(path):
                log(f"skip {tag}")
                continue
            model, best = fit(name, data, len(vocab), seed=seed, epochs=a.epochs, log=log)
            res = {"model": name, "seed": seed, "best_val": best}
            probs = {}
            for sp in [k for k in ["val", "test_iid", "test_ood"] if k in data]:
                pa, ps = predict(model, data[sp])
                res[sp] = evaluate_probs(data[sp], pa, ps)
                probs[f"{sp}_aspect"], probs[f"{sp}_sent"] = pa, ps
            if name in FUSION:  # robustness: what if one modality is missing at test time?
                for drop in ["text", "image"]:
                    pa, ps = predict(model, data["test_iid"], drop=drop)
                    res[f"test_iid_missing_{drop}"] = evaluate_probs(data["test_iid"], pa, ps)
            np.savez_compressed(os.path.join(a.out, "runs", tag + "_probs.npz"), **probs)
            json.dump(res, open(path, "w"), indent=1)
            if name == "cross" and seed == 0 and not a.sentiment_only:
                torch.save(model.state_dict(), "checkpoints/cross_seed0.pt")
            log(f"DONE {tag} test_iid aspF1 {res['test_iid']['aspect']['f1']:.3f} "
                f"sentF1 {res['test_iid']['sentiment']['f1']:.3f} sentAcc {res['test_iid']['sentiment']['acc']:.3f}")


if __name__ == "__main__":
    main()
