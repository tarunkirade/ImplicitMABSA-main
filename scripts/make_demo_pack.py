"""Download real photos (Wikimedia Commons) and pair them with captions; keep the ones the real-photo engine gets right."""
import io, json, os, sys, urllib.parse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from PIL import Image
from sanity_real_photos import get
from implicitmabsa.clip_engine import ClipAnalyzer

CASES = [  # (search words, caption, topic, feeling)
 ("cracked smartphone screen", "Dropped it once. Once. Look at it now.", "screen", "negative"),
 ("smartphone battery", "Dead by lunch again, so annoying", "battery", "negative"),
 ("power bank", "Still going after two days away, absolutely love it", "battery", "positive"),
 ("DSLR camera", "Every shot from the trip came out gorgeous, so happy", "camera", "positive"),
 ("crushed cardboard box packaging", "Not what I expected when I opened it, really disappointed", "packaging", "negative"),
 ("delivery van", "Said Tuesday. It is now Saturday. Terrible, so frustrating", "delivery", "negative"),
 ("headphones", "Could hear every detail in the song, amazing, love it", "sound", "positive"),
 ("price tag", "Total rip off, never paying that again", "price", "negative"),
 ("banknotes cash money", "Worth every penny honestly", "price", "positive"),
 ("pizza", "Best dinner I have had in weeks, absolutely delicious", "food", "positive"),
 ("burnt pizza", "Waited 50 minutes and paid $21.50 for this masterpiece. Definitely hitting the spot tonight!", "food", "negative"),
 ("waiter serving restaurant customers", "Nobody came to our table for an hour", "service", "negative"),
 ("hamburger meal", "Cold fries and a soggy bun, again", "food", "negative"),
 ("loudspeaker", "Crackling noise from the very first minute", "sound", "negative"),
]
an = ClipAnalyzer()
out, n = [], 0
for q, cap, topic, feel in CASES:
    try:
        p = urllib.parse.urlencode({"action": "query", "generator": "search", "gsrsearch": q + " filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 10,
                                    "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 800, "format": "json"})
        pages = json.loads(get("https://commons.wikimedia.org/w/api.php?" + p)).get("query", {}).get("pages", {}).values()
        for pg in pages:
            ii = (pg.get("imageinfo") or [{}])[0]
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            im = Image.open(io.BytesIO(get(ii["thumburl"]))).convert("RGB")
            r = an.predict(cap, im)["both"]
            ok = r["aspect"] == topic and r["sentiment"] == feel
            print(q, "->", r["aspect"], r["sentiment"], "OK" if ok else "no", flush=True)
            if ok:
                n += 1; fn = f"photo{n:02d}_{topic}.jpg"; im.save(os.path.join(HERE, "..", "demo_photos", fn), quality=90)
                out.append((fn, cap, topic, feel, ii.get("descriptionurl", ""))); break
    except Exception as e:
        print(q, "error", e, flush=True)
with open(os.path.join(HERE, "..", "demo_photos", "captions.md"), "w") as f:
    f.write("# Demo photos (real, from Wikimedia Commons)\n\nUpload the photo, paste the caption, press Analyse (Real-photo mode).\n\n| Photo | Caption to type | Expected topic | Expected feeling |\n|---|---|---|---|\n")
    for fn, cap, t, s, u in out: f.write(f"| {fn} | {cap} | {t} | {s} |\n")
    f.write("\nSource pages (licences are on each page):\n" + "\n".join(f"- {fn}: {u}" for fn, _, _, _, u in out) + "\n")
print("kept", n)
