"""Procedural image renderer for ImplicitMABSA-Synth.

Each image can carry two kinds of cue, mirroring real social-media posts:
  * an ASPECT cue  - an object icon (battery, phone screen, camera, price tag, truck, box, speaker, gauge)
  * a SENTIMENT cue - the icon's *state* (full/empty battery, cracked/vivid screen, crushed/neat box, ...)
                      or, when no icon is present, a reaction face.
Icons are drawn at 128x128 with random position / scale / rotation / background / noise and
down-sampled to 64x64 (anti-aliased).  No external assets or pretrained models are needed.
"""
import math
import random
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import numpy as np

GREEN = (40, 167, 69)
RED = (220, 53, 69)
YELLOW = (240, 173, 30)
GREY = (140, 140, 145)
DARK = (35, 38, 45)
WHITE = (250, 250, 250)
BLUE = (52, 120, 220)
BROWN = (176, 132, 84)

STATE_COLOR = {"positive": GREEN, "negative": RED, "neutral": YELLOW, "amb": GREY}
S = 128  # working canvas size


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


# ---------------------------------------------------------------- icons (drawn on RGBA layer)
def _battery(d, st, rng):
    x0, y0, x1, y1 = 22, 44, 100, 84
    d.rounded_rectangle([x0, y0, x1, y1], 6, outline=DARK, width=4)
    d.rectangle([x1 + 1, 55, x1 + 8, 73], fill=DARK)
    frac = {"positive": rng.uniform(.85, 1.0), "neutral": rng.uniform(.4, .6),
            "negative": rng.uniform(.04, .14), "amb": rng.uniform(.2, .8)}[st]
    w = int((x1 - x0 - 10) * frac)
    d.rectangle([x0 + 5, y0 + 5, x0 + 5 + max(w, 3), y1 - 5], fill=STATE_COLOR[st])
    if st == "positive":  # bolt
        d.polygon([(64, 48), (54, 66), (62, 66), (58, 80), (72, 60), (64, 60)], fill=WHITE)


def _screen(d, st, rng):
    d.rounded_rectangle([40, 14, 88, 114], 8, fill=DARK)
    sx0, sy0, sx1, sy1 = 45, 24, 83, 104
    if st == "positive":
        for i in range(sy1 - sy0):
            t = i / (sy1 - sy0)
            d.line([(sx0, sy0 + i), (sx1, sy0 + i)], fill=(int(60 + 190 * t), int(200 - 90 * t), int(255 - 150 * t)))
    elif st == "neutral":
        d.rectangle([sx0, sy0, sx1, sy1], fill=(238, 222, 150))
    elif st == "negative":
        d.rectangle([sx0, sy0, sx1, sy1], fill=(30, 30, 34))
        cx, cy = rng.randint(55, 73), rng.randint(45, 80)
        for _ in range(7):
            a = rng.uniform(0, 2 * math.pi)
            r = rng.randint(18, 40)
            d.line([(cx, cy), (cx + r * math.cos(a), cy + r * math.sin(a))], fill=(235, 235, 235), width=2)
    else:
        d.rectangle([sx0, sy0, sx1, sy1], fill=(120, 120, 128))
    d.ellipse([60, 106, 68, 112], fill=GREY)


def _camera(d, st, rng):
    d.rounded_rectangle([16, 38, 112, 100], 10, fill=(70, 74, 84))
    d.rectangle([34, 30, 58, 40], fill=(70, 74, 84))
    d.ellipse([64 - 25, 69 - 25, 64 + 25, 69 + 25], fill=DARK)
    if st == "positive":
        d.ellipse([64 - 17, 69 - 17, 64 + 17, 69 + 17], fill=BLUE)
        d.ellipse([64 - 9, 69 - 9, 64 + 9, 69 + 9], fill=(120, 190, 255))
        d.ellipse([56, 61, 63, 68], fill=WHITE)
    elif st == "neutral":
        d.ellipse([64 - 17, 69 - 17, 64 + 17, 69 + 17], fill=(110, 120, 140))
    elif st == "negative":
        d.ellipse([64 - 17, 69 - 17, 64 + 17, 69 + 17], fill=(20, 20, 24))
        for _ in range(5):
            a = rng.uniform(0, 2 * math.pi)
            d.line([(64, 69), (64 + 17 * math.cos(a), 69 + 17 * math.sin(a))], fill=(200, 200, 200), width=2)
    else:
        d.ellipse([64 - 17, 69 - 17, 64 + 17, 69 + 17], fill=(90, 90, 96))
    d.ellipse([98, 46, 106, 54], fill=STATE_COLOR[st])


def _price(d, st, rng):
    d.polygon([(20, 40), (76, 40), (100, 64), (76, 88), (20, 88)], fill=(245, 236, 210), outline=DARK)
    d.ellipse([28, 58, 38, 68], fill=DARK)
    d.text((50, 64), "$", fill=DARK, font=_font(40), anchor="mm")
    if st == "positive":  # down arrow
        d.polygon([(100, 98), (120, 98), (110, 118)], fill=GREEN)
        d.rectangle([106, 80, 114, 98], fill=GREEN)
    elif st == "negative":  # up arrow
        d.polygon([(100, 30), (120, 30), (110, 10)], fill=RED)
        d.rectangle([106, 30, 114, 48], fill=RED)
    elif st == "neutral":
        d.rectangle([98, 92, 122, 97], fill=YELLOW)
        d.rectangle([98, 103, 122, 108], fill=YELLOW)


def _delivery(d, st, rng):
    col = STATE_COLOR[st]
    d.rectangle([28, 46, 80, 88], fill=(215, 220, 230), outline=DARK, width=3)
    d.polygon([(80, 56), (100, 56), (112, 72), (112, 88), (80, 88)], fill=(215, 220, 230), outline=DARK)
    for cx in (46, 96):
        d.ellipse([cx - 8, 82, cx + 8, 98], fill=DARK)
    if st == "positive":
        for y in (52, 64, 76):
            d.line([(4, y), (24, y)], fill=GREEN, width=3)
        d.ellipse([96, 22, 122, 48], fill=GREEN)
        d.line([(102, 36), (108, 42), (117, 28)], fill=WHITE, width=3)
    elif st == "negative":
        d.ellipse([96, 20, 124, 48], fill=RED)
        d.line([(110, 26), (110, 34)], fill=WHITE, width=3)
        d.line([(110, 34), (116, 38)], fill=WHITE, width=3)
        d.text((60, 30), "!", fill=RED, font=_font(30), anchor="mm")
    elif st == "neutral":
        for x in (98, 108, 118):
            d.ellipse([x - 3, 34, x + 3, 40], fill=YELLOW)
    else:
        d.ellipse([100, 28, 116, 44], outline=GREY, width=3)


def _packaging(d, st, rng):
    if st == "negative":  # crushed, skewed, torn
        d.polygon([(26, 56), (60, 42), (104, 52), (98, 96), (60, 108), (24, 92)], fill=BROWN, outline=DARK)
        d.line([(30, 70), (48, 64), (56, 78), (74, 70), (92, 84)], fill=DARK, width=3)
        d.polygon([(60, 42), (72, 30), (80, 48)], fill=(150, 108, 66))
        d.line([(98, 96), (86, 90), (92, 78)], fill=RED, width=3)
    else:
        col = {"positive": (198, 156, 106), "neutral": BROWN, "amb": (170, 170, 175)}[st]
        d.rectangle([26, 44, 102, 104], fill=col, outline=DARK, width=3)
        d.rectangle([26, 44, 102, 58], fill=tuple(int(c * .85) for c in col), outline=DARK, width=3)
        if st == "positive":
            d.rectangle([58, 44, 70, 104], fill=GREEN)
            d.polygon([(64, 44), (44, 28), (56, 42)], fill=GREEN)
            d.polygon([(64, 44), (84, 28), (72, 42)], fill=GREEN)
            for (sx, sy) in [(110, 40), (18, 34), (112, 90)]:
                d.line([(sx - 6, sy), (sx + 6, sy)], fill=YELLOW, width=3)
                d.line([(sx, sy - 6), (sx, sy + 6)], fill=YELLOW, width=3)
        elif st == "neutral":
            d.rectangle([58, 44, 70, 58], fill=(214, 190, 140))


def _sound(d, st, rng):
    d.rectangle([16, 52, 36, 78], fill=DARK)
    d.polygon([(36, 52), (60, 32), (60, 98), (36, 78)], fill=DARK)
    if st == "positive":
        for r in (16, 28, 40):
            d.arc([60 - r + 4, 65 - r, 60 + r + 4, 65 + r], -50, 50, fill=GREEN, width=5)
    elif st == "negative":
        pts = [(72 + i * 8, 65 + (18 if i % 2 else -18)) for i in range(6)]
        d.line(pts, fill=RED, width=5)
        d.line([(66, 24), (116, 106)], fill=RED, width=5)
    elif st == "neutral":
        d.arc([60 - 16 + 4, 65 - 16, 60 + 16 + 4, 65 + 16], -50, 50, fill=YELLOW, width=5)
    else:
        for r in (16, 28):
            d.arc([60 - r + 4, 65 - r, 60 + r + 4, 65 + r], -50, 50, fill=GREY, width=5)


def _performance(d, st, rng):
    box = [14, 30, 114, 130]
    for a0, a1, c in [(180, 240, RED), (240, 300, YELLOW), (300, 360, GREEN)]:
        if st == "amb":
            c = GREY
        d.arc(box, a0, a1, fill=c, width=12)
    ang = {"positive": rng.uniform(-25, -8), "neutral": rng.uniform(-100, -80),
           "negative": rng.uniform(-172, -155), "amb": rng.uniform(-100, -80)}[st]
    cx, cy = 64, 80
    d.line([(cx, cy), (cx + 42 * math.cos(math.radians(ang)), cy + 42 * math.sin(math.radians(ang)))],
           fill=DARK, width=5)
    d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=DARK)
    if st != "amb":
        d.ellipse([54, 100, 74, 120], fill=STATE_COLOR[st])


ICONS = {"battery": _battery, "screen": _screen, "camera": _camera, "price": _price,
         "delivery": _delivery, "packaging": _packaging, "sound": _sound, "performance": _performance}


def _face(d, st, rng):
    col = {"positive": (120, 200, 110), "negative": (235, 110, 100), "neutral": (235, 215, 130)}[st]
    d.ellipse([22, 22, 106, 106], fill=col, outline=DARK, width=4)
    d.ellipse([44, 50, 54, 62], fill=DARK)
    d.ellipse([74, 50, 84, 62], fill=DARK)
    if st == "positive":
        d.arc([40, 56, 88, 96], 20, 160, fill=DARK, width=5)
    elif st == "negative":
        d.arc([40, 72, 88, 112], 200, 340, fill=DARK, width=5)
    else:
        d.line([(46, 84), (82, 84)], fill=DARK, width=5)


def _clutter(d, rng):
    """Generic non-informative product-ish scene."""
    for _ in range(rng.randint(2, 4)):
        c = tuple(rng.randint(90, 190) for _ in range(3))
        x, y = rng.randint(15, 95), rng.randint(15, 95)
        w, h = rng.randint(14, 34), rng.randint(14, 34)
        k = rng.random()
        if k < .4:
            d.ellipse([x, y, x + w, y + h], fill=c)
        elif k < .8:
            d.rectangle([x, y, x + w, y + h], fill=c)
        else:
            d.polygon([(x, y + h), (x + w // 2, y), (x + w, y + h)], fill=c)


def _background(rng):
    c1 = tuple(rng.randint(185, 250) for _ in range(3))
    c2 = tuple(rng.randint(185, 250) for _ in range(3))
    arr = np.zeros((S, S, 3), np.float32)
    t = np.linspace(0, 1, S)[:, None, None]
    arr[:] = (1 - t) * np.array(c1) + t * np.array(c2)
    im = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    for _ in range(rng.randint(0, 4)):  # low-contrast distractors
        x, y, r = rng.randint(0, S), rng.randint(0, S), rng.randint(6, 26)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 28))
    return im


def render(aspect, state, mode, seed):
    """Render one image.

    aspect: one of ICONS keys or None
    state : 'positive' | 'negative' | 'neutral' | 'amb' (icon state) or face sentiment
    mode  : 'icon'   -> aspect icon in `state`   (aspect cue, sentiment cue iff state != 'amb')
            'face'   -> reaction face in `state` (sentiment cue only)
            'none'   -> clutter                  (no cue)
    """
    rng = random.Random(seed)
    bg = _background(rng)
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if mode == "icon":
        ICONS[aspect](d, state, rng)
    elif mode == "face":
        _face(d, state, rng)
    else:
        _clutter(d, rng)
    scale = rng.uniform(.8, 1.05)
    layer = layer.resize((int(S * scale), int(S * scale)), Image.BILINEAR)
    layer = layer.rotate(rng.uniform(-10, 10), resample=Image.BILINEAR, expand=False)
    ox = rng.randint(-10, 10) + (S - layer.size[0]) // 2
    oy = rng.randint(-10, 10) + (S - layer.size[1]) // 2
    bg.paste(layer, (ox, oy), layer)
    arr = np.asarray(bg).astype(np.float32)
    arr += np.random.RandomState(seed % (2 ** 31)).normal(0, 6, arr.shape)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < .15:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(.6, 1.4)))
    return im.resize((64, 64), Image.LANCZOS)
