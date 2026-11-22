"""Text-only, image-only and multimodal models for implicit aspect + sentiment classification.

All models are trained from scratch (no pretrained weights are required), and every model has two
heads: an aspect head (8 classes) and a sentiment head (3 classes).

    text          BiGRU + attention pooling
    image         4-block CNN + global average pooling
    concat        feature concatenation -> MLP                        (simple early fusion baseline)
    cross         Cross-modal attention + gated fusion + aspect-aware sentiment head  (proposed)
    cross_noasp   proposed model without the aspect-conditioned sentiment head          (ablation)
    cross_nomd    proposed model trained without modality dropout                       (ablation)
(the late-fusion baseline averages the probabilities of the independently trained text and image models.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD

N_ASPECT, N_SENT = 8, 3


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, emb=96, hid=96, word_drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hid, 1)
        self.word_drop = word_drop
        self.out_dim = 2 * hid

    def forward(self, tok):
        mask = tok != PAD
        if self.training and self.word_drop > 0:  # randomly blank words -> robustness to unseen wording
            drop = (torch.rand_like(tok, dtype=torch.float) < self.word_drop) & mask
            tok = tok.masked_fill(drop, 1)  # UNK
        h, _ = self.gru(self.emb(tok))
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        w = F.softmax(a, dim=-1)
        pooled = (h * w.unsqueeze(-1)).sum(1)
        return h, mask, pooled


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class ImageEncoder(nn.Module):
    def __init__(self, ch=(16, 32, 64, 64)):
        super().__init__()
        self.net = nn.Sequential(_block(3, ch[0]), _block(ch[0], ch[1]), _block(ch[1], ch[2]), _block(ch[2], ch[3]))
        self.out_dim = ch[3]

    def forward(self, img):
        fmap = self.net(img)                       # B,C,4,4
        regions = fmap.flatten(2).transpose(1, 2)  # B,16,C
        return regions, regions.mean(1)


def _heads(d, hidden=128, p=0.2, cond=False):
    asp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, N_ASPECT))
    sent = nn.Sequential(nn.Linear(d + (N_ASPECT if cond else 0), hidden), nn.ReLU(), nn.Dropout(p),
                         nn.Linear(hidden, N_SENT))
    return asp, sent


class TextOnly(nn.Module):
    def __init__(self, vocab_size, **kw):
        super().__init__()
        self.t = TextEncoder(vocab_size)
        self.asp, self.sent = _heads(self.t.out_dim)

    def forward(self, tok, img, drop=None):
        _, _, p = self.t(tok)
        return self.asp(p), self.sent(p)


class ImageOnly(nn.Module):
    def __init__(self, vocab_size=None, **kw):
        super().__init__()
        self.v = ImageEncoder()
        self.asp, self.sent = _heads(self.v.out_dim)

    def forward(self, tok, img, drop=None):
        _, p = self.v(img)
        return self.asp(p), self.sent(p)


class Concat(nn.Module):
    def __init__(self, vocab_size, mod_drop=0.0, **kw):
        super().__init__()
        self.t, self.v = TextEncoder(vocab_size), ImageEncoder()
        d = self.t.out_dim + self.v.out_dim
        self.asp, self.sent = _heads(d)
        self.mod_drop = mod_drop

    def forward(self, tok, img, drop=None):
        _, _, pt = self.t(tok)
        _, pv = self.v(img)
        if drop == "text":
            pt = torch.zeros_like(pt)
        elif drop == "image":
            pv = torch.zeros_like(pv)
        f = torch.cat([pt, pv], -1)
        return self.asp(f), self.sent(f)


class CrossModal(nn.Module):
    """Proposed model.

    1. project word states and image-region features into a shared d-dim space
    2. text->image cross-attention (words look at image regions) and image->text cross-attention
    3. gated fusion  f = g * text_ctx + (1-g) * image_ctx  (g learned from both modalities)
    4. aspect head on f; sentiment head conditioned on the predicted aspect distribution
       (aspect-aware sentiment: the polarity is about *that* implicit aspect)
    5. modality dropout at training time so the model can cope with an uninformative modality
    """

    def __init__(self, vocab_size, d=96, heads=4, cond_aspect=True, mod_drop=0.15, **kw):
        super().__init__()
        self.t, self.v = TextEncoder(vocab_size), ImageEncoder()
        self.pt = nn.Linear(self.t.out_dim, d)
        self.pv = nn.Linear(self.v.out_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, 16, d))
        self.t2v = nn.MultiheadAttention(d, heads, batch_first=True)
        self.v2t = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_t, self.ln_v = nn.LayerNorm(d), nn.LayerNorm(d)
        self.gate = nn.Linear(4 * d, d)
        self.asp, self.sent = _heads(d + 2 * d, cond=cond_aspect)
        self.cond_aspect, self.mod_drop, self.d = cond_aspect, mod_drop, d
        self.last_attn = None

    def forward(self, tok, img, drop=None):
        ht, mask, _ = self.t(tok)
        regions, _ = self.v(img)
        T = self.pt(ht)
        V = self.pv(regions) + self.pos
        B = T.size(0)
        if self.training and self.mod_drop > 0 and drop is None:
            r = torch.rand(B, 1, 1, device=T.device)
            T = T * (r >= self.mod_drop / 2)          # drop text for ~7.5% of samples
            V = V * ((r < self.mod_drop / 2) | (r >= self.mod_drop))  # drop image for ~7.5%
        elif drop == "text":
            T = torch.zeros_like(T)
        elif drop == "image":
            V = torch.zeros_like(V)
        # text words attend to image regions; image regions attend to words
        tv, attn = self.t2v(T, V, V, need_weights=True)
        vt, _ = self.v2t(V, T, T, key_padding_mask=~mask)
        self.last_attn = attn.detach()                # B, L, 16  (word -> region attention)
        tv = self.ln_t(T + tv)
        vt = self.ln_v(V + vt)
        m = mask.unsqueeze(-1).float()
        t_ctx = (tv * m).sum(1) / m.sum(1).clamp(min=1)
        v_ctx = vt.mean(1)
        g = torch.sigmoid(self.gate(torch.cat([t_ctx, v_ctx, t_ctx * v_ctx, (t_ctx - v_ctx).abs()], -1)))
        fused = g * t_ctx + (1 - g) * v_ctx
        f = torch.cat([fused, t_ctx, v_ctx], -1)
        a_logits = self.asp(f)
        if self.cond_aspect:
            s_logits = self.sent(torch.cat([f, F.softmax(a_logits, -1)], -1))
        else:
            s_logits = self.sent(f)
        return a_logits, s_logits


def build(name, vocab_size):
    if name == "text":
        return TextOnly(vocab_size)
    if name == "image":
        return ImageOnly()
    if name == "concat":
        return Concat(vocab_size)
    if name == "cross":
        return CrossModal(vocab_size)
    if name == "cross_noasp":
        return CrossModal(vocab_size, cond_aspect=False)
    if name == "cross_nomd":
        return CrossModal(vocab_size, mod_drop=0.0)
    raise ValueError(name)
