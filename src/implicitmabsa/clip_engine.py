"""Real-photo engine: zero-shot analysis with a pretrained CLIP model (no training on our data).

Why this exists
  The trained cross-modal model only ever saw the small generated icons, so a real photograph (a phone with a
  broken screen, say) is outside what it knows.  CLIP was pretrained on hundreds of millions of real photo-caption
  pairs, so it can read real photos and free-form captions.

How it works
  topic    photo  -> CLIP image embedding compared with short descriptions of what each topic looks like
           caption-> CLIP text embedding compared with short descriptions of each topic
           together = the two scores multiplied (product of experts); the caption counts as much as it is informative
  feeling  photo  -> CLIP image embedding vs. "broken / happy / ordinary product" descriptions
           caption-> VADER lexicon (social-media sentiment) with a neutral band
           together = the two multiplied
  Everything below (prompts, weights) is a transparent heuristic, chosen on the VALIDATION split only.

    from implicitmabsa.clip_engine import ClipAnalyzer
    r = ClipAnalyzer().predict("why does this happen to me only", "my_phone.jpg")
    r["both"]["aspect"], r["both"]["sentiment"]
"""
import math
import os

from .lexicon import ASPECTS, SENTIMENTS

# what each topic looks like in a photo
IMAGE_PROMPTS = {
    "battery": ["a phone with a low battery warning", "a charging cable plugged into a phone", "a power bank charging a device",
                "a dead phone that will not turn on", "a battery icon"],
    "screen": ["a cracked phone screen", "a phone with a glitching or corrupted display", "a bright colourful display screen",
               "a monitor or tablet screen", "a touchscreen"],
    "camera": ["a camera", "a camera lens", "a photograph taken with a phone camera", "a blurry photo", "a photographer taking a picture"],
    "price": ["a price tag", "cash money and coins", "a shopping receipt", "a sale discount sign", "a wallet with money"],
    "delivery": ["a delivery van", "a parcel waiting on a doorstep", "a courier handing over a package",
                 "a package tracking screen", "a delivery truck on the road"],
    "packaging": ["a cardboard box", "a damaged crushed parcel", "product packaging", "a sealed box with bubble wrap",
                  "an unboxing of a new product"],
    "sound": ["a loudspeaker", "a pair of headphones", "a microphone", "sound waves", "a music speaker"],
    "performance": ["a laptop running slowly", "a speedometer gauge", "a benchmark chart", "a loading spinner", "a fast sports car"],
}
# extra topics for objects outside the gadget/shopping list (real-photo mode only; the trained model has the first 8)
IMAGE_PROMPTS.update({
    "food": ["a plate of cooked food", "a pizza", "a restaurant meal", "burnt or undercooked food", "a burger and fries", "a takeaway meal in a box"],
    "service": ["a long queue of people waiting", "a waiter serving customers", "a clock showing a long wait", "an angry customer at a counter",
                "a customer service desk"],
    "other": ["a landscape", "a person smiling", "an animal", "a building", "an abstract picture or meme"],
})
EXTRA_TOPICS = ["food", "service", "other"]
ALL_TOPICS = list(ASPECTS) + EXTRA_TOPICS

# what a caption about each topic is about
TEXT_PROMPTS = {
    "battery": "a post about battery life and charging",
    "screen": "a post about the display and screen",
    "camera": "a post about the camera and photo quality",
    "price": "a post about price and value for money",
    "delivery": "a post about shipping and delivery",
    "packaging": "a post about the box and packaging",
    "sound": "a post about audio and speaker sound",
    "performance": "a post about speed and performance",
    "food": "a post about the food and how it tastes",
    "service": "a post about waiting time and staff service",
    "other": "a post that is not about a product",
}
# what a photo shows when the feeling is negative / neutral / positive (order = SENTIMENTS)
SENT_PROMPTS = [
    ["a broken, damaged or defective product", "a cracked or malfunctioning device", "an error message", "a disappointed unhappy customer",
     "burnt, spoiled or ruined food", "a messy, dirty or unappetizing meal"],
    ["an ordinary product photo", "a plain neutral picture of an object", "an average everyday item"],
    ["a brand new product in perfect condition", "a happy satisfied customer", "a thumbs up", "a smiling person with a new gadget",
     "delicious freshly cooked food"],
]

# tuned on the validation split (see scripts/tune_clip.py)
DEFAULTS = dict(w_text=0.6, k_sent=3.0, neutral_bias=0.8, w_cap_sent=1.0, img_t=1.0,
                sarcasm_cap_pos=0.6, sarcasm_img_neg=0.5, sarcasm_w_cap=0.05)


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def caption_sentiment_probs(text, k=DEFAULTS["k_sent"], nb=DEFAULTS["neutral_bias"], _vader=[None]):
    """VADER compound score -> [neg, neu, pos] probabilities with a neutral band."""
    if _vader[0] is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _vader[0] = SentimentIntensityAnalyzer()
    c = _vader[0].polarity_scores(text)["compound"]
    return _softmax([-k * c, nb, k * c])


class ClipAnalyzer:
    def __init__(self, model_name="ViT-B-32-quickgelu", pretrained="openai", topics=None, **params):
        import open_clip
        import torch

        self.torch = torch
        self.p = {**DEFAULTS, **params}
        self.model, _, self.pre = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tok = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.topics = list(topics or ALL_TOPICS)
        self.asp_prompts = [p for a in self.topics for p in IMAGE_PROMPTS[a]]
        self.sen_prompts = [p for g in SENT_PROMPTS for p in g]       # pass topics=ASPECTS for the 8-topic benchmark setting
        self.asp_img, self.asp_slices = self._bank([p for a in self.topics for p in IMAGE_PROMPTS[a]], [len(IMAGE_PROMPTS[a]) for a in self.topics])
        self.asp_txt = self.txt([TEXT_PROMPTS[a] for a in self.topics])
        self.sen_img, self.sen_slices = self._bank([p for g in SENT_PROMPTS for p in g], [len(g) for g in SENT_PROMPTS])

    # ---- embeddings
    def txt(self, strings, bs=128):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(strings), bs):
                f = self.model.encode_text(self.tok(strings[i:i + bs]))
                out.append(f / f.norm(dim=-1, keepdim=True))
        return self.torch.cat(out)

    def img(self, images, bs=64):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), bs):
                x = self.torch.stack([self.pre(im.convert("RGB")) for im in images[i:i + bs]])
                f = self.model.encode_image(x)
                out.append(f / f.norm(dim=-1, keepdim=True))
        return self.torch.cat(out)

    def _bank(self, prompts, sizes):
        sl, s = [], 0
        for n in sizes:
            sl.append((s, s + n))
            s += n
        return self.txt(prompts), sl

    def _group_max(self, sim, slices):
        return self.torch.stack([sim[:, a:b].max(1).values for a, b in slices], 1)

    # ---- scores from features   (all return python lists of probabilities, one row per item)
    def aspect_from_image(self, fi):
        lg = 100 * self._group_max(fi @ self.asp_img.T, self.asp_slices) * self.p["img_t"]
        return lg.softmax(-1)

    def aspect_from_text(self, ft):
        return (100 * ft @ self.asp_txt.T).softmax(-1)

    def sentiment_from_image(self, fi):
        return (100 * self._group_max(fi @ self.sen_img.T, self.sen_slices)).softmax(-1)

    def sentiment_from_text(self, texts):
        return self.torch.tensor([caption_sentiment_probs(t, self.p["k_sent"], self.p["neutral_bias"]) for t in texts])

    def combine(self, p_img, p_txt, w):
        """Product of experts:  p ~ p_img * p_txt^w   (rows are probability vectors)."""
        lg = p_img.clamp_min(1e-9).log() + w * p_txt.clamp_min(1e-9).log()
        return lg.softmax(-1)

    def analyse_features(self, fi, ft, texts):
        a_i, a_t = self.aspect_from_image(fi), self.aspect_from_text(ft)
        s_i, s_t = self.sentiment_from_image(fi), self.sentiment_from_text(texts)
        a_b = self.combine(a_i, a_t, self.p["w_text"])
        # sarcasm check: the caption sounds positive but the photo clearly looks bad -> trust the photo more
        sarcasm = (s_t[:, 2] > self.p["sarcasm_cap_pos"]) & (s_i[:, 0] > self.p["sarcasm_img_neg"])
        w = self.torch.where(sarcasm, self.torch.tensor(self.p["sarcasm_w_cap"]), self.torch.tensor(self.p["w_cap_sent"]))
        lg = s_i.clamp_min(1e-9).log() + w.unsqueeze(1) * s_t.clamp_min(1e-9).log()
        s_b = lg.softmax(-1)
        self.last_sarcasm = sarcasm.tolist()
        return {"both": (a_b, s_b), "text": (a_t, s_t), "image": (a_i, s_i)}

    # ---- single prediction (same output shape as api.Analyzer.predict)
    def predict(self, text, image):
        from PIL import Image

        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        fi, ft = self.img([image]), self.txt([text])
        res, out = self.analyse_features(fi, ft, [text]), {}
        for name, (a, s) in res.items():
            a, s = a[0].tolist(), s[0].tolist()
            ia, is_ = max(range(len(a)), key=a.__getitem__), max(range(len(s)), key=s.__getitem__)
            out[name] = {"aspect": self.topics[ia], "aspect_conf": a[ia], "aspect_probs": dict(zip(self.topics, a)),
                         "sentiment": SENTIMENTS[is_], "sentiment_conf": s[is_], "sentiment_probs": dict(zip(SENTIMENTS, s))}
        out["tokens"], out["unknown_tokens"], out["sarcasm"] = text.lower().split(), [], bool(self.last_sarcasm[0])
        return out


def _explain(self, text, image):
    """Live evidence for the decision: which photo cells / words matter (occlusion), and which description matched the photo."""
    from PIL import Image, ImageStat

    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    image = image.convert("RGB")
    fi0, ft0 = self.img([image]), self.txt([text])
    a0, s0 = self.analyse_features(fi0, ft0, [text])["both"]
    ca, cs = int(a0.argmax()), int(s0.argmax())
    n = 3
    grey = tuple(int(v) for v in ImageStat.Stat(image).mean)
    W, H = image.size
    cells = []
    for r in range(n):
        for c in range(n):
            q = image.copy()
            q.paste(grey, (c * W // n, r * H // n, (c + 1) * W // n, (r + 1) * H // n))
            cells.append(q)
    fi = self.img(cells)
    oa, os_ = self.analyse_features(fi, ft0.repeat(len(cells), 1), [text] * len(cells))["both"]
    grid = lambda v, base: [[float(base - v[r * n + c]) for c in range(n)] for r in range(n)]  # noqa: E731
    words, wl = [], text.split()[:30]
    if len(wl) > 1:
        variants = [" ".join(wl[:j] + wl[j + 1:]) for j in range(len(wl))]
        wa, ws = self.analyse_features(fi0.repeat(len(wl), 1), self.txt(variants), variants)["both"]
        words = [{"w": w, "aspect": float(a0[0, ca] - wa[j, ca]), "sentiment": float(s0[0, cs] - ws[j, cs])} for j, w in enumerate(wl)]
    elif wl:
        words = [{"w": wl[0], "aspect": 0.0, "sentiment": 0.0}]
    # which description of the winning topic / feeling the photo matched best
    sims = (fi0 @ self.asp_img.T)[0]
    lo, hi = self.asp_slices[ca]
    ia = lo + int(sims[lo:hi].argmax())
    sims_s = (fi0 @ self.sen_img.T)[0]
    lo2, hi2 = self.sen_slices[cs]
    is_ = lo2 + int(sims_s[lo2:hi2].argmax())
    return {"grid": n, "aspect_drop": grid(oa[:, ca], a0[0, ca]), "sentiment_drop": grid(os_[:, cs], s0[0, cs]), "words": words,
            "matched_topic_prompt": self.asp_prompts[ia], "matched_topic_sim": float(sims[ia]),
            "matched_feeling_prompt": self.sen_prompts[is_], "matched_feeling_sim": float(sims_s[is_]),
            "model": "pretrained CLIP ViT-B/32 (zero-shot)", "params": int(sum(p.numel() for p in self.model.parameters()))}


ClipAnalyzer.explain = _explain


def evaluate(an, rows, data_dir=None):
    """Same output shape as api.evaluate, for the CLIP engine."""
    from PIL import Image

    from .api import DATA, image_path, prf

    data_dir = data_dir or DATA
    rows = [r for r in rows if image_path(r, data_dir)]
    fi = an.img([Image.open(image_path(r, data_dir)) for r in rows])
    ft = an.txt([r["text"] for r in rows])
    ya = [ASPECTS.index(r["aspect"]) for r in rows]
    ys = [SENTIMENTS.index(r["sentiment"]) for r in rows]
    out = {}
    for name, (a, s_) in an.analyse_features(fi, ft, [r["text"] for r in rows]).items():
        pa, ps = a.argmax(1).tolist(), s_.argmax(1).tolist()
        out[name] = {"aspect": prf(ya, pa, 8), "sentiment": prf(ys, ps, 3),
                     "joint": 100 * sum(x == y and u == v for x, y, u, v in zip(ya, pa, ys, ps)) / max(len(rows), 1)}
    return {"n": len(rows), "result": out}
