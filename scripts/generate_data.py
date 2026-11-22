"""Generate the ImplicitMABSA-Synth dataset:  python scripts/generate_data.py --out data"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from implicitmabsa.datagen import generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    _, stats = generate(a.out, seed=a.seed)
    print(json.dumps({k: v["n"] for k, v in stats.items()}))
