from app.schemas import Segment
from app.vtt import format_vtt_timestamp, segments_to_vtt


def test_format_vtt_timestamp():
    assert format_vtt_timestamp(0) == "00:00:00.000"
    assert format_vtt_timestamp(1.5) == "00:00:01.500"
    assert format_vtt_timestamp(65.04) == "00:01:05.040"
    assert format_vtt_timestamp(3723.456) == "01:02:03.456"


def test_segments_to_vtt_shape():
    segs = [
        Segment(start=0.0, end=1.2, text="hello"),
        Segment(start=1.2, end=2.5, text="world"),
    ]
    vtt = segments_to_vtt(segs, header_note="test")
    assert vtt.startswith("WEBVTT\n")
    assert "NOTE test" in vtt
    assert "00:00:00.000 --> 00:00:01.200" in vtt
    assert "hello" in vtt
    assert "00:00:01.200 --> 00:00:02.500" in vtt
    assert "world" in vtt


def test_transcribe_includes_vtt(client, silent_wav_bytes):
    res = client.post(
        "/transcribe?lang=en",
        files={"file": ("clip.wav", silent_wav_bytes, "audio/wav")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["alignment"] == "ctc"
    assert body["vtt"].startswith("WEBVTT")
    assert "-->" in body["vtt"]
    assert body["vttPath"]
    assert len(body["segments"]) >= 2
    assert body["segments"][0]["start"] < body["segments"][0]["end"]
