import torch
from implicitmabsa.models import build

def test_forward_shapes_and_missing_modality():
    tok = torch.randint(2, 50, (4, 24)); tok[:, 10:] = 0
    img = torch.randn(4, 3, 64, 64)
    for n in ["text", "image", "concat", "cross", "cross_noasp", "cross_nomd"]:
        m = build(n, 60).eval()
        a, s = m(tok, img)
        assert a.shape == (4, 8) and s.shape == (4, 3)
        if n in ("concat", "cross", "cross_noasp", "cross_nomd"):
            for d in ("text", "image"):
                a, s = m(tok, img, drop=d)
                assert torch.isfinite(a).all() and torch.isfinite(s).all()
