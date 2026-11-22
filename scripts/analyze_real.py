"""Tables + figure for the real-data (Twitter-2015/2017) experiment. python scripts/analyze_real.py"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = json.load(open("results/real/real_results.json"))
NAMES = {"majority": "Majority class", "text": "Text-only", "image": "Image-only", "late": "Late fusion",
         "concat": "Concat fusion", "cross": "Cross-modal"}
COL = {"majority": "#8a8a85", "text": "#2a78d6", "image": "#eb6834", "late": "#1baf7a", "concat": "#eda100", "cross": "#e87ba4"}
CONDS = [("2015_explicit", "Twitter-2015\ntarget shown"), ("2015_masked", "Twitter-2015\ntarget masked"),
         ("2017_explicit", "Twitter-2017\ntarget shown"), ("2017_masked", "Twitter-2017\ntarget masked")]
CONDS = [c for c in CONDS if c[0] in R]
f = lambda v: f"{v[0]:.1f}±{v[1]:.1f}" if isinstance(v, list) else f"{v:.1f}"
md = []
for k, lab in CONDS:
    md += [f"### {lab.replace(chr(10), ' – ')}", "", "| Model | Acc | Precision | Recall | Macro-F1 |", "|---|---|---|---|---|"]
    for n in NAMES:
        r = R[k][n]
        md.append(f"| {NAMES[n]} | " + " | ".join(f(r[m]) for m in ("acc", "precision", "recall", "f1")) + " |")
    md.append("")
open("results/real/real_tables.md", "w").write("\n".join(md)); print("\n".join(md))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
for ax, met, ttl in zip(axes, ["acc", "f1"], ["Accuracy (%)", "Macro-F1 (%)"]):
    w = 0.8 / len(NAMES)
    for i, n in enumerate(NAMES):
        vals = [R[k][n][met] for k, _ in CONDS]
        m = [v[0] if isinstance(v, list) else v for v in vals]
        e = [v[1] if isinstance(v, list) else 0 for v in vals]
        ax.bar(np.arange(len(CONDS)) + (i - (len(NAMES) - 1) / 2) * w, m, w * .9, yerr=e, color=COL[n], label=NAMES[n],
               capsize=1.2, error_kw=dict(lw=.6))
    ax.set_xticks(range(len(CONDS)), [c[1] for c in CONDS], fontsize=7)
    ax.set_title(ttl, loc="left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig("results/figures/fig_real.png", bbox_inches="tight", dpi=150)
