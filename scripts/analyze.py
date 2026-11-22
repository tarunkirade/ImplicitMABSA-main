"""Aggregate runs -> tables (markdown/CSV/LaTeX-ish) and figures.   python scripts/analyze.py"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from implicitmabsa.lexicon import ASPECTS, SENTIMENTS  # noqa: E402

RES, FIG = "results", "results/figures"
os.makedirs(FIG, exist_ok=True)
DATA = "data"
ORDER = ["text", "image", "late", "concat", "cross"]
NAMES = {"text": "Text-only", "image": "Image-only", "late": "Late fusion", "concat": "Concat fusion",
         "cross": "Cross-modal (ours)", "cross_noasp": "Ours w/o aspect-conditioning",
         "cross_nomd": "Ours w/o modality dropout"}
# validated categorical slots 1..5 (fixed order): blue, orange, aqua, yellow, magenta
COL = {"text": "#2a78d6", "image": "#eb6834", "late": "#1baf7a", "concat": "#eda100", "cross": "#e87ba4"}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.titlesize": 10, "figure.dpi": 150, "savefig.bbox": "tight"})

rows = list(csv.DictReader(open(f"{DATA}/metadata.csv", encoding="utf-8")))
by = {k: [r for r in rows if r["split"] == k] for k in ["train", "val", "test_iid", "test_ood"]}
y = {k: (np.array([ASPECTS.index(r["aspect"]) for r in v]), np.array([SENTIMENTS.index(r["sentiment"]) for r in v]))
     for k, v in by.items()}

runs = {}
for p in sorted(glob.glob(f"{RES}/runs/*_probs.npz")):
    tag = os.path.basename(p)[:-10]
    name, seed = tag.rsplit("_s", 1)
    runs.setdefault(name, {})[int(seed)] = dict(np.load(p))
# late fusion = mean of unimodal probabilities
if "text" in runs and "image" in runs:
    runs["late"] = {s: {k: (runs["text"][s][k] + runs["image"][s][k]) / 2 for k in runs["text"][s]}
                    for s in runs["text"] if s in runs["image"]}


def mets(yt, pr):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    p = pr.argmax(1)
    P, R, F, _ = precision_recall_fscore_support(yt, p, average="macro", zero_division=0)
    return dict(acc=accuracy_score(yt, p), precision=P, recall=R, f1=F)


def agg(name, split):
    out = {}
    ya, ys = y[split]
    per = []
    for s, d in runs[name].items():
        ma, ms = mets(ya, d[f"{split}_aspect"]), mets(ys, d[f"{split}_sent"])
        joint = float(((d[f"{split}_aspect"].argmax(1) == ya) & (d[f"{split}_sent"].argmax(1) == ys)).mean())
        per.append((ma, ms, joint))
    for task, i in (("aspect", 0), ("sentiment", 1)):
        for m in ("acc", "precision", "recall", "f1"):
            v = np.array([p[i][m] for p in per]) * 100
            out[f"{task}_{m}"] = (v.mean(), v.std())
    v = np.array([p[2] for p in per]) * 100
    out["joint_acc"] = (v.mean(), v.std())
    out["n_seeds"] = len(per)
    return out


table = {sp: {n: agg(n, sp) for n in list(NAMES) if n in runs} for sp in ["test_iid", "test_ood"]}
json.dump({sp: {n: {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()} for n, d in t.items()}
           for sp, t in table.items()}, open(f"{RES}/summary.json", "w"), indent=1)

fmt = lambda t: f"{t[0]:.1f} ± {t[1]:.1f}"
md = []
for sp, title in (("test_iid", "IID test (seen phrasings)"), ("test_ood", "OOD test (unseen phrasings)")):
    md += [f"### {title}", "",
           "| Model | Asp Acc | Asp P | Asp R | Asp F1 | Sent Acc | Sent P | Sent R | Sent F1 | Joint Acc |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for n, d in table[sp].items():
        md.append(f"| {NAMES[n]} | " + " | ".join(fmt(d[k]) for k in (
            "aspect_acc", "aspect_precision", "aspect_recall", "aspect_f1",
            "sentiment_acc", "sentiment_precision", "sentiment_recall", "sentiment_f1", "joint_acc")) + " |")
    md.append("")

# ---- subgroup analysis on IID test: accuracy by where the cue lives
sub = {}
src_a = np.array([r["aspect_src"] for r in by["test_iid"]])
src_s = np.array([r["sent_src"] for r in by["test_iid"]])
conf = np.array([int(r["conflict"]) for r in by["test_iid"]])
ya, ys = y["test_iid"]
for n in [m for m in ORDER if m in runs]:
    accs_a = {g: [] for g in "TIB"}
    accs_s = {g: [] for g in "TIB"}
    accs_c = []
    for s, d in runs[n].items():
        pa, ps = d["test_iid_aspect"].argmax(1), d["test_iid_sent"].argmax(1)
        for g in "TIB":
            accs_a[g].append(((pa == ya) & (src_a == g)).sum() / (src_a == g).sum() * 100)
            accs_s[g].append(((ps == ys) & (src_s == g)).sum() / (src_s == g).sum() * 100)
        accs_c.append(((ps == ys) & (conf == 1)).sum() / max((conf == 1).sum(), 1) * 100)
    sub[n] = dict(aspect={g: float(np.mean(v)) for g, v in accs_a.items()},
                  sent={g: float(np.mean(v)) for g, v in accs_s.items()}, conflict=float(np.mean(accs_c)))
json.dump(sub, open(f"{RES}/subgroups.json", "w"), indent=1)
md += ["### Accuracy (%) by where the cue is available (IID test, mean over seeds)", "",
       "| Model | Aspect: text-only cue | image-only cue | both | Sentiment: text-only cue | image-only cue | both | Text-image conflict |",
       "|---|---|---|---|---|---|---|---|"]
for n, d in sub.items():
    md.append(f"| {NAMES[n]} | " + " | ".join(f"{d['aspect'][g]:.1f}" for g in "TIB") + " | " +
              " | ".join(f"{d['sent'][g]:.1f}" for g in "TIB") + f" | {d['conflict']:.1f} |")
md.append("")

# ---- missing-modality robustness (fusion models)
md += ["### Missing-modality test (IID test; F1 %, mean over seeds)", "",
       "| Model | Both | Text missing (asp/sent) | Image missing (asp/sent) |", "|---|---|---|---|"]
miss = {}
for n in ["concat", "cross", "cross_noasp", "cross_nomd"]:
    fs = sorted(glob.glob(f"{RES}/runs/{n}_s?.json"))
    if not fs:
        continue
    J = [json.load(open(f)) for f in fs]
    g = lambda k, t: np.mean([j[k][t]["f1"] for j in J]) * 100
    miss[n] = {k: (g(k, "aspect"), g(k, "sentiment")) for k in ("test_iid", "test_iid_missing_text", "test_iid_missing_image")}
    m = miss[n]
    md.append(f"| {NAMES[n]} | {m['test_iid'][0]:.1f}/{m['test_iid'][1]:.1f} | "
              f"{m['test_iid_missing_text'][0]:.1f}/{m['test_iid_missing_text'][1]:.1f} | "
              f"{m['test_iid_missing_image'][0]:.1f}/{m['test_iid_missing_image'][1]:.1f} |")
json.dump(miss, open(f"{RES}/missing_modality.json", "w"), indent=1)
open(f"{RES}/results_tables.md", "w").write("\n".join(md))
print("\n".join(md))

# ================= FIGURES =================
# 1. main comparison (F1 per model: aspect & sentiment, IID vs OOD)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, sp, ttl in zip(axes, ["test_iid", "test_ood"], ["IID test (seen wording)", "OOD test (unseen wording)"]):
    ms = [m for m in ORDER if m in table[sp]]
    x = np.arange(2)
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        vals = [table[sp][m]["aspect_f1"], table[sp][m]["sentiment_f1"]]
        ax.bar(x + (i - (len(ms) - 1) / 2) * w, [v[0] for v in vals], w * 0.9, yerr=[v[1] for v in vals],
               color=COL[m], label=NAMES[m], capsize=1.5, error_kw=dict(lw=.7, ecolor=MUTED))
        for xi, v in zip(x, vals):
            ax.text(xi + (i - (len(ms) - 1) / 2) * w, v[0] + v[1] + 1, f"{v[0]:.0f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x, ["Implicit aspect", "Sentiment"])
    ax.set_title(ttl, loc="left")
    ax.set_ylim(0, 108)
axes[0].set_ylabel("Macro-F1 (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, ncol=len(l), loc="lower center", bbox_to_anchor=(0.5, -0.09))
fig.savefig(f"{FIG}/fig_main_f1.png")
plt.close(fig)

# 2. subgroup: where does each modality help
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
lab = {"T": "text-only\ncue", "I": "image-only\ncue", "B": "both\nmodalities"}
for ax, key, ttl in zip(axes, ["aspect", "sent"], ["Implicit aspect accuracy", "Sentiment accuracy"]):
    ms = [m for m in ORDER if m in sub]
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        ax.bar(np.arange(3) + (i - (len(ms) - 1) / 2) * w, [sub[m][key][g] for g in "TIB"], w * 0.9,
               color=COL[m], label=NAMES[m])
    ax.set_xticks(range(3), [lab[g] for g in "TIB"])
    ax.set_title(ttl + " by cue location", loc="left")
axes[0].set_ylabel("Accuracy (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, ncol=len(l), loc="lower center", bbox_to_anchor=(0.5, -0.17))
fig.savefig(f"{FIG}/fig_subgroups.png")
plt.close(fig)

# 3. confusion matrices for the proposed model (seed 0, IID test)
if "cross" in runs:
    d = runs["cross"][min(runs["cross"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[8, 3]))
    for ax, (pk, yt, labs, ttl) in zip(axes, [("test_iid_aspect", ya, ASPECTS, "Implicit aspect"),
                                             ("test_iid_sent", ys, SENTIMENTS, "Sentiment")]):
        cm = confusion_matrix(yt, d[pk].argmax(1), labels=range(len(labs)))
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labs)), labs, rotation=45, ha="right")
        ax.set_yticks(range(len(labs)), labs)
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, f"{cmn[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cmn[i, j] > .55 else INK)
        ax.set_title(ttl + " (row-normalised %)", loc="left")
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    fig.savefig(f"{FIG}/fig_confusion.png")
    plt.close(fig)

# 4. missing-modality robustness
if miss:
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    cats = [("test_iid", "Both"), ("test_iid_missing_image", "Image\nmissing"), ("test_iid_missing_text", "Text\nmissing")]
    ms = [m for m in ["concat", "cross", "cross_nomd"] if m in miss]
    mc = {"concat": COL["concat"], "cross": COL["cross"], "cross_nomd": "#4a3aa7"}
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        vals = [(miss[m][k][0] + miss[m][k][1]) / 2 for k, _ in cats]
        ax.bar(np.arange(3) + (i - (len(ms) - 1) / 2) * w, vals, w * 0.9, color=mc[m], label=NAMES[m])
        for xi, v in zip(np.arange(3), vals):
            ax.text(xi + (i - (len(ms) - 1) / 2) * w, v + 1, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(3), [c[1] for c in cats])
    ax.set_ylabel("Mean of aspect & sentiment F1 (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.set_title("Robustness when a modality is missing", loc="left")
    fig.savefig(f"{FIG}/fig_missing_modality.png")
    plt.close(fig)

# 5. qualitative examples (IID test): where does fusion fix unimodal errors?
if all(k in runs for k in ("text", "image", "cross")):
    from PIL import Image
    s0 = min(runs["cross"])
    P = {n: (runs[n][s0]["test_iid_aspect"].argmax(1), runs[n][s0]["test_iid_sent"].argmax(1)) for n in ("text", "image", "cross")}
    ok = lambda n, i: (P[n][0][i] == ya[i], P[n][1][i] == ys[i])
    picks = [i for i in range(len(ya)) if ok("cross", i) == (True, True) and not all(ok("text", i)) and not all(ok("image", i))]
    rng = np.random.RandomState(0)
    picks = list(rng.choice(picks, size=min(6, len(picks)), replace=False)) if picks else []
    if picks:
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
        for ax, i in zip(axes.ravel(), picks):
            r = by["test_iid"][i]
            ax.imshow(Image.open(f"{DATA}/{r['image']}"))
            ax.axis("off")
            f = lambda n: f"{ASPECTS[P[n][0][i]]}/{SENTIMENTS[P[n][1][i]][:3]}"
            t = r["text"] if len(r["text"]) < 44 else r["text"][:42] + "…"
            ax.set_title(f"“{t}”\ntruth: {r['aspect']}/{r['sentiment'][:3]}\ntext: {f('text')}  image: {f('image')}\nours: {f('cross')}",
                         fontsize=6.3, loc="left")
        fig.tight_layout()
        fig.savefig(f"{FIG}/fig_examples.png")
        plt.close(fig)

# 6. dataset overview: dataset examples + composition
from implicitmabsa.imagegen import ICONS, render  # noqa: E402
fig, axes = plt.subplots(3, 8, figsize=(7.2, 3.1))
for c, a in enumerate(ICONS):
    for r, s in enumerate(["negative", "neutral", "positive"]):
        axes[r, c].imshow(render(a, s, "icon", 500 + c * 7 + r))
        axes[r, c].axis("off")
    axes[0, c].set_title(a, fontsize=7)
for r, s in enumerate(["negative", "neutral", "positive"]):
    axes[r, 0].text(-0.15, 0.5, s[:3], transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
fig.savefig(f"{FIG}/fig_dataset_icons.png")
plt.close(fig)
print("figures written")
