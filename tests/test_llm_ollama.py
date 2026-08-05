import json
import socket
import sys
import urllib.error
from unittest.mock import MagicMock, patch

from darija_overrides import llm_ollama


def _fake_response(payload: dict):
    cm = MagicMock()
    cm.__enter__.return_value = MagicMock(read=lambda: None)
    cm.read = lambda: json.dumps(payload).encode("utf-8")
    return cm


def test_fix_arabic_json_punctuation_makes_json_loads_succeed():
    broken = '{"a":"x"،"b":"y"}'
    assert llm_ollama._fix_arabic_json_punctuation(broken) == '{"a":"x","b":"y"}'
    json.loads(llm_ollama._fix_arabic_json_punctuation(broken))


def test_strip_trailing_commas_makes_json_loads_succeed():
    broken = '{"highlights": [{"a": 1,}, {"b": 2,},],}'
    fixed = llm_ollama._strip_trailing_commas(broken)
    assert fixed == '{"highlights": [{"a": 1}, {"b": 2}]}'
    json.loads(fixed)


def test_salvage_truncated_highlights_recovers_complete_elements():
    # Real shape captured from Atlas-Chat-9B: 2 complete highlight objects
    # followed by a third cut off mid-string, no closing at all.
    truncated = (
        '{"highlights":[{"title":"t1","start_time":0.0,"end_time":10.0,"score":80,'
        '"hook_sentence":"h1","virality_reason":"r1"},'
        '{"title":"t2","start_time":10.0,"end_time":40.0,"score":70,'
        '"hook_sentence":"h2","virality_reason":"r2"},'
        '{"title":"t3 incomplete","start_time":40.0,"end_time":90.0,"score":60,'
        '"hook_sentence":"cut off mid strin'
    )
    salvaged = llm_ollama._salvage_truncated_highlights(truncated)
    parsed = json.loads(salvaged)
    assert [h["title"] for h in parsed["highlights"]] == ["t1", "t2"]


def test_salvage_truncated_highlights_no_complete_element_returns_unchanged():
    truncated = '{"highlights":[{"title":"only one, cut off mid strin'
    assert llm_ollama._salvage_truncated_highlights(truncated) == truncated


def test_salvage_truncated_highlights_leaves_wellformed_json_parseable():
    wellformed = (
        '{"highlights":[{"title":"t1","start_time":0.0,"end_time":10.0,'
        '"score":80,"hook_sentence":"h1","virality_reason":"r1"}]}'
    )
    parsed = json.loads(llm_ollama._salvage_truncated_highlights(wellformed))
    assert len(parsed["highlights"]) == 1


def test_salvage_truncated_highlights_is_noop_without_a_bracket():
    text = '{"content_type": "podcast", "density": "medium"}'
    assert llm_ollama._salvage_truncated_highlights(text) == text


def test_salvage_truncated_highlights_stops_at_repeated_highlights_key():
    # Real shape captured from Atlas-Chat-9B: it repeated "highlights" as
    # several sibling keys instead of one array with several elements.
    # json.loads on that (if left as-is) silently keeps only the *last*
    # key's value — salvage must stop at the first clean array close
    # instead, so nothing gets silently dropped without at least an error.
    weird = (
        '{"highlights":[{"title":"a","start_time":0,"end_time":68,"score":90,'
        '"hook_sentence":"h","virality_reason":"r"}],'
        '"highlights":[{"title":"b","start_time":68,"end_time":147,"score":85,'
        '"hook_sentence":"h2","virality_reason":"r2"}],'
        '"highlights":[{"title":"c cut off mid stri'
    )
    salvaged = llm_ollama._salvage_truncated_highlights(weird)
    parsed = json.loads(salvaged)
    assert [h["title"] for h in parsed["highlights"]] == ["a"]


def test_call_local_llm_sends_prompt_and_returns_normalized_text():
    with patch("darija_overrides.llm_ollama.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = MagicMock(
            read=lambda: b'{"response": "{\\"a\\":\\"x\\"\\u060c\\"b\\":\\"y\\"}"}'
        )
        result = llm_ollama.call_local_llm("some prompt")

    request_arg = mock_urlopen.call_args[0][0]
    sent_payload = json.loads(request_arg.data)
    assert sent_payload["prompt"] == "some prompt"
    assert sent_payload["model"] == llm_ollama.OLLAMA_MODEL
    assert sent_payload["stream"] is False
    assert sent_payload["keep_alive"] == 0
    assert result == '{"a":"x","b":"y"}'
    json.loads(result)


def test_call_local_llm_salvages_a_truncated_highlights_response():
    truncated_response = (
        '{"highlights":[{"title":"t1","start_time":0.0,"end_time":10.0,"score":80,'
        '"hook_sentence":"h1","virality_reason":"r1"},'
        '{"title":"t2 cut off mid stri'
    )
    with patch("darija_overrides.llm_ollama.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = MagicMock(
            read=lambda: json.dumps({"response": truncated_response}).encode("utf-8")
        )
        result = llm_ollama.call_local_llm("some prompt")

    parsed = json.loads(result)
    assert [h["title"] for h in parsed["highlights"]] == ["t1"]


def test_call_local_llm_model_override():
    with patch("darija_overrides.llm_ollama.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = MagicMock(
            read=lambda: b'{"response": "ok"}'
        )
        llm_ollama.call_local_llm("prompt", model="some-other-model")

    request_arg = mock_urlopen.call_args[0][0]
    sent_payload = json.loads(request_arg.data)
    assert sent_payload["model"] == "some-other-model"


def test_call_local_llm_returns_empty_string_on_timeout_instead_of_raising():
    with patch("darija_overrides.llm_ollama.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = socket.timeout("timed out")
        result = llm_ollama.call_local_llm("some prompt")
    assert result == ""


def test_call_local_llm_returns_empty_string_on_connection_error():
    with patch("darija_overrides.llm_ollama.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        result = llm_ollama.call_local_llm("some prompt")
    assert result == ""


def test_install_shadows_vendor_llm_module():
    sys.modules.pop("shorts_generator.local.llm", None)
    try:
        llm_ollama.install()
        assert sys.modules["shorts_generator.local.llm"] is llm_ollama
        assert (
            sys.modules["shorts_generator.local.llm"].call_local_llm
            is llm_ollama.call_local_llm
        )
    finally:
        sys.modules.pop("shorts_generator.local.llm", None)
