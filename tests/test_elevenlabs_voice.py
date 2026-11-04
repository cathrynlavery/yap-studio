"""Speech generation sets an HTTP timeout below its process deadline."""

import json

import pytest
from voice_helpers import calls

from yap_studio import elevenlabs, tools


@pytest.mark.parametrize("page", [
    {"voices": [], "has_more": True, "next_page_token": None},
    {"voices": [], "has_more": True, "next_page_token": "again"},
    {"voices": "unreadable", "has_more": False, "next_page_token": None},
])
def test_personal_voices_refuses_incomplete_pages(fake_voice, monkeypatch, page):
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_PAGES", json.dumps([page]))
    with pytest.raises(elevenlabs.GenerationError, match="unreadable voices"):
        elevenlabs.personal_voices("My voice")
    assert len(calls(fake_voice)) == (10 if page["next_page_token"] == "again" else 1)


def test_personal_voices_accepts_bare_list(fake_voice, monkeypatch):
    monkeypatch.setenv("FAKE_ELEVENLABS_RAW", "1")
    assert [row["name"] for row in elevenlabs.personal_voices("My voice")] == ["My voice"]


def test_speech_request_timeout(tmp_path, fake_voice, monkeypatch):
    run_tool = tools.run_tool
    deadlines = []

    def run(name, args, **kwargs):
        if args[0] == "text-to-speech":
            deadlines.append(kwargs["timeout_s"])
        return run_tool(name, args, **kwargs)

    monkeypatch.setattr(tools, "run_tool", run)
    target = tmp_path / "speech.mp3"
    elevenlabs.generate_voice(
        "fake-personal-voice-secret",
        {"text": "cold plunge", "previous_text": "", "next_text": "", "seed": 1}, target,
    )
    argv = calls(fake_voice)[0]
    assert argv[argv.index("--timeout") + 1] == "2m"
    assert deadlines == [4 * 120 + 60]
    assert target.stat().st_size > 0
