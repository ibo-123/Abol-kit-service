from app.asr import MmsAsrEngine, UnsupportedLangError
from app.schemas import LANG_TO_MMS, MMS_TO_APP


def test_lang_mapping_am_en():
    assert LANG_TO_MMS["am"] == "amh"
    assert LANG_TO_MMS["en"] == "eng"
    assert MMS_TO_APP["amh"] == "am"
    assert MMS_TO_APP["eng"] == "en"


def test_resolve_mms_lang():
    assert MmsAsrEngine.resolve_mms_lang("am") == "amh"
    assert MmsAsrEngine.resolve_mms_lang("EN") == "eng"


def test_resolve_mms_lang_rejects_unknown():
    try:
        MmsAsrEngine.resolve_mms_lang("fr")
        assert False, "expected UnsupportedLangError"
    except UnsupportedLangError:
        pass
