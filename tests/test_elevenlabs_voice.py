"""Speech generation sets an HTTP timeout below its process deadline."""

from voice_helpers import calls

from yap_studio import elevenlabs, tools


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
