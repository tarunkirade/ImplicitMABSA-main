"""Models for REAL Twitter data on frozen pretrained features (sentiment only, 3 classes).

Text : frozen spaCy word vectors -> BiGRU -> attention pooling
Image: frozen MobileNetV3 3x3 region grid (960-d)
Same five systems as the synthetic study: text, image, late fusion (avg probs), concat, cross-modal attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEnc(nn.Module):
    def __init__(self, emb_matrix, hid=128, p=0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix), freeze=True, padding_idx=0)
        self.gru = nn.GRU(emb_matrix.shape[1], hid, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hid, 1)
        self.drop = nn.Dropout(p)
        self.out_dim = 2 * hid

    def forward(self, ids):
        mask = ids != 0
        h, _ = self.gru(self.drop(self.emb(ids)))
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        w = F.softmax(a, -1)
        return h, mask, (h * w.unsqueeze(-1)).sum(1)


class ImgEnc(nn.Module):
    def __init__(self, d=256, p=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(960)
        self.proj = nn.Sequential(nn.Dropout(p), nn.Linear(960, d), nn.ReLU())
        self.out_dim = d

    def forward(self, g):                       # B,9,960
        r = self.proj(self.ln(g))
        return r, r.mean(1)


def head(d, p=0.3):
    return nn.Sequential(nn.Dropout(p), nn.Linear(d, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, 3))


class TextOnly(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.t = TextEnc(emb)
        self.h = head(self.t.out_dim)

    def forward(self, ids, g, drop=None):
        return self.h(self.t(ids)[2])


class ImageOnly(nn.Module):
    def __init__(self, emb=None):
        super().__init__()
        self.v = ImgEnc()
        self.h = head(self.v.out_dim)

    def forward(self, ids, g, drop=None):
        return self.h(self.v(g)[1])


class Concat(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.t, self.v = TextEnc(emb), ImgEnc()
        self.h = head(self.t.out_dim + self.v.out_dim)

    def forward(self, ids, g, drop=None):
        return self.h(torch.cat([self.t(ids)[2], self.v(g)[1]], -1))


class Cross(nn.Module):
    def __init__(self, emb, d=128, heads=4):
        super().__init__()
        self.t, self.v = TextEnc(emb), ImgEnc()
        self.pt, self.pv = nn.Linear(self.t.out_dim, d), nn.Linear(self.v.out_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, 9, d))
        self.t2v = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        self.v2t = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        self.lt, self.lv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.gate = nn.Linear(4 * d, d)
        self.h = head(3 * d)

    def forward(self, ids, g, drop=None):
        ht, mask, _ = self.t(ids)
        r, _ = self.v(g)
        T, V = self.pt(ht), self.pv(r) + self.pos
        tv, _ = self.t2v(T, V, V)
        vt, _ = self.v2t(V, T, T, key_padding_mask=~mask)
        tv, vt = self.lt(T + tv), self.lv(V + vt)
        m = mask.unsqueeze(-1).float()
        tc, vc = (tv * m).sum(1) / m.sum(1).clamp(min=1), vt.mean(1)
        gt = torch.sigmoid(self.gate(torch.cat([tc, vc, tc * vc, (tc - vc).abs()], -1)))
        return self.h(torch.cat([gt * tc + (1 - gt) * vc, tc, vc], -1))


def build(name, emb):
    return {"text": TextOnly, "image": ImageOnly, "concat": Concat, "cross": Cross}[name](emb)
