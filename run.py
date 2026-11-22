"""ImplicitMABSA - one entry point.  Everything works from the terminal; the web UI is optional.

    python run.py setup                          create the synthetic dataset (once, ~1 min)
    python run.py demo                           analyse a random test sample and print the answer
    python run.py demo --text "Dead by lunch" --image my_photo.jpg
    python run.py evaluate --split test_iid      accuracy / precision / recall / F1 of the trained model
    python run.py real                           print the real-data (Twitter) results
    python run.py ui                             OPTIONAL: open the web app in your browser
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def cmd_setup(a):
    from implicitmabsa.api import count_images
    from implicitmabsa.datagen import generate

    if count_images() >= 8000 and not a.force:
        print("Dataset already created (8,000 images in data/images).")
        return
    print("Creating the synthetic dataset (8,000 image-text pairs) ...")
    generate(os.path.join(ROOT, "data"))
    print("Done.")


def cmd_demo(a):
    from implicitmabsa.api import Analyzer, image_path, load_rows

    a.engine = a.engine or ("clip" if a.image else "trained")

    truth = None
    if a.text and a.image:
        text, img = a.text, a.image
    else:
        rows = [r for r in load_rows(split=a.split) if image_path(r)]
        if not rows:
            sys.exit("No images found. Run:  python run.py setup")
        r = random.Random(a.seed).choice(rows)
        text, img, truth = r["text"], image_path(r), (r["aspect"], r["sentiment"])
    if a.engine == "clip":
        from implicitmabsa.clip_engine import ClipAnalyzer

        print("Loading the pretrained real-photo model (first time: ~350 MB download) ...")
        res = ClipAnalyzer().predict(text, img)
    else:
        res = Analyzer().predict(text, img)
    line = "-" * 62
    print(f'\n  caption : "{text}"\n  image   : {os.path.relpath(str(img), ROOT)}\n  {line}')
    for key, title in (("both", "text + image"), ("text", "text only"), ("image", "image only")):
        r = res[key]
        print(f"  {title:<13} -> about: {r['aspect']:<12} ({r['aspect_conf']:>4.0%})   feeling: {r['sentiment']:<8} ({r['sentiment_conf']:>4.0%})")
    if truth:
        print(f"  {line}\n  correct answer -> about: {truth[0]:<12}          feeling: {truth[1]}")
    print()


def cmd_evaluate(a):
    from implicitmabsa.api import Analyzer, evaluate, image_path, load_rows

    rows = [r for r in load_rows(split=a.split) if image_path(r)][: a.n]
    if len(rows) < 20:
        sys.exit("Not enough images. Run:  python run.py setup")
    if a.engine == "clip":
        from implicitmabsa.clip_engine import ClipAnalyzer, evaluate as clip_evaluate

        print("Loading the pretrained real-photo model ...")
        res = clip_evaluate(ClipAnalyzer(), rows)
    else:
        res = evaluate(Analyzer(), rows)
    print(f"\n  {res['n']} samples from '{a.split}'   (all values in %, macro-averaged)\n")
    print(f"  {'input used':<14}{'Aspect: Acc':>12}{'P':>7}{'R':>7}{'F1':>7}   {'Sentiment: Acc':>14}{'P':>7}{'R':>7}{'F1':>7}")
    for key, title in (("both", "text + image"), ("text", "text only"), ("image", "image only")):
        A, S = res["result"][key]["aspect"], res["result"][key]["sentiment"]
        print(f"  {title:<14}{A['acc']:>12.1f}{A['precision']:>7.1f}{A['recall']:>7.1f}{A['f1']:>7.1f}   "
              f"{S['acc']:>14.1f}{S['precision']:>7.1f}{S['recall']:>7.1f}{S['f1']:>7.1f}")
    print()


def cmd_real(a):
    p = os.path.join(ROOT, "results", "real", "real_results.json")
    if not os.path.exists(p):
        sys.exit("results/real/real_results.json not found")
    R = json.load(open(p, encoding="utf-8"))
    names = [("majority", "majority class"), ("text", "text only"), ("image", "image only"), ("late", "late fusion"),
             ("concat", "concat fusion"), ("cross", "cross-modal")]
    for key, res in R.items():
        year, variant = key.split("_")
        print(f"\n  Twitter-{year}, target {'shown' if variant == 'explicit' else 'masked'}   (sentiment, 3 seeds, %)")
        print(f"  {'model':<16}{'Acc':>7}{'P':>7}{'R':>7}{'F1':>7}")
        for m, n in names:
            if m in res:
                g = lambda k: res[m][k][0] if isinstance(res[m][k], list) else res[m][k]  # noqa: E731
                print(f"  {n:<16}{g('acc'):>7.1f}{g('precision'):>7.1f}{g('recall'):>7.1f}{g('f1'):>7.1f}")
    print()


def cmd_ui(a):
    sys.argv = ["server", "--port", str(a.port)] + (["--no-browser"] if a.no_browser else [])
    sys.path.insert(0, os.path.join(ROOT, "ui"))
    import server

    server.main()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="create the synthetic dataset")
    s.add_argument("--force", action="store_true")
    s.set_defaults(f=cmd_setup)
    s = sub.add_parser("demo", help="analyse one text + image pair")
    s.add_argument("--text")
    s.add_argument("--image")
    s.add_argument("--split", default="test_iid")
    s.add_argument("--seed", type=int, default=None, help="fix the random sample")
    s.add_argument("--engine", choices=["trained", "clip"], default=None,
                   help="trained = model trained on the generated data;  clip = pretrained real-photo mode "
                        "(default: clip when you pass your own --image, otherwise trained)")
    s.set_defaults(f=cmd_demo)
    s = sub.add_parser("evaluate", help="metrics of the trained model on a split")
    s.add_argument("--split", default="test_iid", choices=["val", "test_iid", "test_ood"])
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--engine", choices=["trained", "clip"], default="trained")
    s.set_defaults(f=cmd_evaluate)
    s = sub.add_parser("real", help="print the real-data (Twitter) results")
    s.set_defaults(f=cmd_real)
    s = sub.add_parser("ui", help="optional web app")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(f=cmd_ui)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
