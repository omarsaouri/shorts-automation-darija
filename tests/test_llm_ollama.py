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
