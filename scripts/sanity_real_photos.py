"""Small independent check on REAL photos (needs internet): does the photo alone point to the right topic?

For each topic a few search words are used on Wikimedia Commons; the first photos found are the test images and the
search topic is their label.  It is a sanity check (about 50 photos), NOT a benchmark - labels come from search words.
Compares the trained model (never saw real photos) with the pretrained real-photo engine, photo only.

    python scripts/sanity_real_photos.py          -> results/real_photo_sanity.json
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from implicitmabsa.lexicon import ASPECTS  # noqa: E402

QUERIES = {
    "battery": ["smartphone battery", "power bank", "phone charging cable"],
    "screen": ["cracked smartphone screen", "LCD monitor display", "smartphone touchscreen"],
    "camera": ["digital camera", "camera lens", "DSLR camera"],
    "price": ["price tag", "banknotes cash money", "sale discount sign"],
    "delivery": ["delivery van", "parcel courier delivery", "package delivered doorstep"],
    "packaging": ["cardboard box packaging", "product packaging box", "crushed cardboard box"],
    "sound": ["loudspeaker", "headphones", "studio microphone"],
    "performance": ["speedometer", "sports car racing", "computer processor CPU"],
}
# extra topics (real-photo mode only), to see how it copes with objects outside the gadget list
EXTRA_QUERIES = {
    "food": ["pizza", "hamburger meal", "restaurant dish plate"],
    "service": ["people waiting in queue", "waiter serving restaurant customers"],
    "other": ["mountain landscape", "domestic cat", "portrait of a person"],
}
PER_TOPIC = 6
UA = {"User-Agent": "ImplicitMABSA-sanity/1.0 (student project)"}


def get(url, tries=6):
    for i in range(tries):                       # be polite to Wikimedia: pause, and back off when told to slow down
        time.sleep(3)
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            time.sleep(8 * (i + 1))


def search(q, n=4):
    p = urllib.parse.urlencode({"action": "query", "generator": "search", "gsrsearch": q + " filetype:bitmap", "gsrnamespace": 6,
                                "gsrlimit": n, "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 400, "format": "json"})
    d = json.loads(get("https://commons.wikimedia.org/w/api.php?" + p))
    out = []
    for pg in (d.get("query", {}).get("pages", {}) or {}).values():
        ii = (pg.get("imageinfo") or [{}])[0]
        if ii.get("mime", "").startswith("image/") and ii.get("thumburl"):
            out.append((pg["title"], ii["thumburl"]))
    return out


def main():
    from implicitmabsa.api import Analyzer
    from implicitmabsa.clip_engine import ClipAnalyzer

    photos = []
    for asp, qs in {**QUERIES, **EXTRA_QUERIES}.items():
        got = []
        for q in qs:
            try:
                found = search(q)
            except Exception:  # noqa: BLE001  (rate limited: skip)
                continue
            for title, url in found:
                if len(got) >= PER_TOPIC or any(title == g[0] for g in got):
                    continue
                try:
                    got.append((title, Image.open(io.BytesIO(get(url, tries=2))).convert("RGB")))
                except Exception:  # noqa: BLE001
                    pass
        photos += [(asp, t, im) for t, im in got]
        print(asp, len(got), flush=True)
    done = sorted({a for a, _, _ in photos})
    if len(done) < len(QUERIES) + len(EXTRA_QUERIES):
        print("WARNING: only these topics could be fetched (rate limit):", done)
    trained, clip8, clip11 = Analyzer(), ClipAnalyzer(topics=ASPECTS), ClipAnalyzer()
    rows = []
    for asp, title, im in photos:
        rows.append({"label": asp, "file": title, "trained": trained.predict("ok", im)["image"]["aspect"],
                     "clip8": clip8.predict("ok", im)["image"]["aspect"], "clip11": clip11.predict("ok", im)["image"]["aspect"]})
    core = [r for r in rows if r["label"] in ASPECTS]
    acc = lambda rs, k: 100 * sum(r[k] == r["label"] for r in rs) / max(len(rs), 1)  # noqa: E731
    res = {"topics": done, "n_photos": len(core), "n_extra_photos": len(rows) - len(core), "chance": 100 / len(ASPECTS),
           "accuracy_photo_only": {"trained": acc(core, "trained"), "clip": acc(core, "clip8"), "clip_11_topics": acc(core, "clip11")},
           "accuracy_extra_topics_11": acc([r for r in rows if r["label"] not in ASPECTS], "clip11"), "rows": rows}
    json.dump(res, open(os.path.join(HERE, "..", "results", "real_photo_sanity.json"), "w"), indent=1)
    print(f"{len(core)} gadget photos | photo-only topic accuracy: trained {res['accuracy_photo_only']['trained']:.0f}%  "
          f"real-photo engine (8 topics) {res['accuracy_photo_only']['clip']:.0f}%  (11 topics) {res['accuracy_photo_only']['clip_11_topics']:.0f}%  "
          f"| extra food/service/other photos ({res['n_extra_photos']}): {res['accuracy_extra_topics_11']:.0f}%")


if __name__ == "__main__":
    main()
