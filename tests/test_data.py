"""Data-integrity tests: implicitness, coverage, split hygiene."""
import re
from implicitmabsa import lexicon as L


def _all_phrases():
    for a, d in L.PHRASES.items():
        for k, lst in d.items():
            for p in lst:
                yield a, k, p
    for k, lst in L.GENERIC_SENT.items():
        for p in lst:
            yield None, k, p
    for p in L.FILLER:
        yield None, "filler", p


def test_no_explicit_aspect_terms():
    """No phrase (for ANY aspect) may contain any explicit aspect term -> aspects are implicit."""
    bad = []
    for a, k, p in _all_phrases():
        low = p.lower()
        for asp, terms in L.EXPLICIT_TERMS.items():
            for t in terms:
                if t in low:
                    bad.append((asp, t, p))
    assert not bad, bad


def test_list_sizes():
    for a in L.ASPECTS:
        for s in L.SENTIMENTS:
            assert len(L.PHRASES[a][s]) == 8
        assert len(L.PHRASES[a]["topic"]) == 4
    for s in L.SENTIMENTS:
        assert len(L.GENERIC_SENT[s]) == 8
    assert len(L.FILLER) == 8


def test_phrases_unique():
    seen = [p for _, _, p in _all_phrases()]
    assert len(seen) == len(set(seen))


def test_generated_dataset_properties(tmp_path):
    from implicitmabsa.datagen import generate
    rows, st = generate(str(tmp_path), n_train=300, n_val=60, n_test_iid=60, n_test_ood=60, seed=3)
    assert len(rows) == 480
    for r in rows:
        # every sample must have the aspect AND sentiment recoverable from >=1 modality
        assert r["text_has_aspect"] or r["img_has_aspect"]
        assert r["text_has_sent"] or r["img_has_sent"]
        low = r["text"].lower()
        for terms in L.EXPLICIT_TERMS.values():
            assert not any(t in low for t in terms), r["text"]
    assert (tmp_path / "metadata.csv").exists()
    assert len(list((tmp_path / "images").glob("*.png"))) == 480
    # OOD phrases never leak into train
    train_txt = " ".join(r["text"] for r in rows if r["split"] == "train")
    for a in L.ASPECTS:
        for s in L.SENTIMENTS:
            for p in L.PHRASES[a][s][L.N_SEEN:]:
                assert p not in train_txt
