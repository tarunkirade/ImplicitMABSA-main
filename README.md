# ImplicitMABSA

**Given a photo and a short caption, guess what the post is about and how the writer feels — even when the caption never says what it is about.**

> *"Not what I expected!"* + photo of a crushed parcel → topic: **packaging**, feeling: **negative**

The caption alone is ambiguous; the photo alone does not say how the writer feels. Together they give the answer. This project builds a model that uses both, and compares it with a text-only and an image-only model (accuracy, precision, recall, F1) — as asked in the brief (PJ_05).

---

## Run it — pick whichever you like

**Windows: double-click `run.bat`.** The first time it installs what it needs and creates the dataset; then you get a menu:

| Choice | What it does |
|---|---|
| 1 | Opens the web app in your browser (**optional**). Click **Quit app** (top right) to stop it and come back to this menu |
| 2 | Analyses a random example, printed in the window |
| 3 | Measures the trained model: accuracy, precision, recall, F1 |
| 4 | Prints the real-data (Twitter) results |

**Terminal (any system), no UI at all:**

```
pip install -r requirements-ui.txt
python run.py setup                          # create the dataset once (~1 min)
python run.py demo                           # analyse a random example
python run.py demo --text "Dead by lunch again" --image my_photo.jpg
python run.py evaluate --split test_ood      # metrics on a test set
python run.py real                           # real-data results
python run.py ui                             # optional web app
```

**From your own Python code:**

```python
from implicitmabsa.api import Analyzer          # run with  PYTHONPATH=src
result = Analyzer().predict("Dead by lunch again", "data/sample_images/test_iid_00001.png")
result["both"]["aspect"], result["both"]["sentiment"]     # caption + photo together
result["text"], result["image"]                           # caption only / photo only
```

---

## Two engines — pick the one that fits your photo

| Engine | Best for | What it is |
|---|---|---|
| **Real-photo mode** (default for uploads) | Any real photograph, any caption | A **pretrained CLIP** model reads the photo and the caption; a social-media sentiment lexicon (VADER) reads the feeling of the caption. Zero-shot — not trained on this project's data. Topics: the 8 product topics plus **food**, **service** and **something else**. If the caption sounds positive but the photo looks bad it flags possible sarcasm and trusts the photo more. The first use downloads ~350 MB. |
| **Trained model** | The generated examples | The cross-modal network trained in this project on the generated dataset. It has only ever seen small generated icons, so it is **not** reliable on real photographs. |

`python run.py demo --image my_photo.jpg --text "why does this happen to me only"` uses real-photo mode automatically.

## The data (read this — it is the honest part)

| | What | Used for |
|---|---|---|
| **Generated benchmark** (ImplicitMABSA-Synth) | 8,000 caption + image pairs made by a script (`scripts/generate_data.py`). The topic is never written in the caption; clues are spread between caption and photo. 8 topics × 3 feelings. | **Topic + feeling.** The only place implicit topics can be scored, because no public dataset has implicit topics + images + feelings together. |
| **Real tweets** (Twitter-2015 / 2017) | Genuine tweets with images, from the public `TwitterDataMABSA` repository. | **Feeling only** (real tweets have no topic labels). Used to check the conclusions on real data. |

Scores on the generated data show that the method works, **not** how it would do in the real world.
The `data/real/` folder holds 800 real tweets (small thumbnails) with each model's predictions, so the app can show them without a big download.

## Results (macro-F1 %, mean of 3 runs)

| Model | Topic · familiar wording | Feeling · familiar | Topic · new wording | Feeling · new |
|---|---|---|---|---|
| Caption only | 75.1 | 79.7 | 17.1 | 45.1 |
| Photo only | 69.7 | 73.4 | 71.2 | 75.3 |
| Caption + photo (concat) | 100.0 | 98.2 | 72.6 | 72.6 |
| Caption + photo (cross-modal, ours) | 100.0 | 98.0 | 68.5 | 65.7 |

**In plain words**
- Using caption **and** photo beats either alone — on the generated data this is by design, since each holds part of the clue.
- The fancier cross-modal model is **not** better than simple concatenation.
- With wording the model never saw, the caption-only model collapses; those that use the photo hold up better.
- On **real tweets** the caption does most of the work; the photo adds no reliable gain.

## Can the scores be better? (CLIP heads)

`scripts/extract_clip_features.py` + `scripts/run_clip_heads.py` train small layers on top of **frozen** CLIP features (3 runs, macro-F1 %, results in `results/clip_heads.json`):

| Model | Topic · new wording | Feeling · new wording |
|---|---|---|
| Caption only | 48.8 | 54.4 |
| Photo only | 71.8 | 73.8 |
| Caption + photo (late) | 87.0 | 81.2 |
| (earlier from-scratch cross-modal model) | 68.5 | 65.7 |

Familiar wording stays at ~100 / ~97. On **real tweets** it does *not* help the caption model (about 57 vs 62 F1); only the photo-alone model improves (about 49 vs 39).

## How you can see the model really decides

In the app, **Show the evidence** appears under every live answer: time taken, number of parameters, a heatmap of which photo areas and which words mattered (hide one part, re-run, see the answer change). There is no table of stored answers; change a word or photo and the answer changes. The *Real tweets* view is the only precomputed one, and says so.

## How the model works

1. **Caption** → words → a small recurrent network (BiGRU) with attention.
2. **Photo** → a small convolutional network that outputs a 4×4 grid of patch features.
3. **Fusion** → words look at the patches that matter and the patches look back at the words (cross-modal attention); a gate decides how much to trust each side.
4. **Two answers** → the topic (8 classes), then the feeling (3 classes) given that topic.

Baselines: text-only, image-only, late fusion (average the two answers), concat fusion.

## What is where

```
run.bat / run.py        start here (menu / terminal commands)
src/implicitmabsa/
    api.py              the simple interface used by everything (predict, evaluate)
    clip_engine.py      real-photo mode (pretrained CLIP, zero-shot)
    models.py           the networks          train.py     training loop
    datagen.py          makes the generated dataset (+ lexicon.py, imagegen.py)
    real_models.py      models for the real tweets
ui/                     the optional web app (server.py + static/index.html)
scripts/                run_experiments.py (train all models), analyze.py, run_real.py, extract_features.py, build_real_demo.py,
                        tune_clip.py (settings of real-photo mode, chosen on validation), sanity_real_photos.py
data/                   generated dataset (+ real/ = real-tweet sample)
checkpoints/            the trained model the app and demo use
results/                tables and figures behind the paper
tests/                  python -m pytest
docs/                   builds the paper
```

Re-run everything from scratch: `python scripts/run_experiments.py` (train), `python scripts/analyze.py` (tables + figures).
Real-data pipeline: `extract_features.py` → `run_real.py` → `build_real_demo.py`.

## Limitations
- Sarcasm is only handled by a simple rule (positive caption + bad-looking photo); sarcasm without such a photo is read literally. More topics can be added in `clip_engine.py`, but each extra topic costs a little accuracy on the others (79% → 75% on the check photos).
- Real-photo mode is a zero-shot heuristic (CLIP + prompts), not a trained model. It scores about 50 F1 on the generated benchmark (those drawn icons are not photos) — see Results. Judge it on real photos.
- A phone photo can legitimately relate to several topics (screen, battery, camera…); the caption should disambiguate. A caption that says nothing about the topic leaves the photo to decide.
- The generated benchmark is easier and cleaner than real posts; its absolute numbers are not real-world claims.
- Real-data experiments use frozen pretrained features and cover feeling only.
- The text model is trained from scratch (small vocabulary), so it struggles with unseen wording.
