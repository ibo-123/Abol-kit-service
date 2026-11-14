from app.align import pack_cues
from app.schemas import Segment


def test_pack_cues_merges_short_words():
    words = [
        Segment(start=0.0, end=0.4, text="a"),
        Segment(start=0.4, end=0.8, text="b"),
        Segment(start=0.8, end=1.2, text="c"),
    ]
    cues = pack_cues(words, duration_s=1.2, max_chars=48, max_seconds=5.0)
    assert len(cues) == 1
    assert cues[0].text == "a b c"
    assert cues[0].start == 0.0


def test_pack_cues_splits_on_max_chars():
    words = [
        Segment(start=0.0, end=1.0, text="abcdefghij"),
        Segment(start=1.0, end=2.0, text="klmnopqrst"),
        Segment(start=2.0, end=3.0, text="uvwxyz1234"),
    ]
    cues = pack_cues(words, duration_s=3.0, max_chars=15, max_seconds=10.0)
    assert len(cues) >= 2
    assert all(c.end > c.start for c in cues)
