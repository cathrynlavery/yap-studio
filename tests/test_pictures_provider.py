"""Offline WaveSpeed reads, redaction and recovery contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_stock_provider import calls

from yap_studio import process, tools
from yap_studio.affiliate import ELEVENLABS_AFFILIATE_URL, WAVESPEED_AFFILIATE_URL
from yap_studio.errors import DependencyError
from yap_studio.pictures import wavespeed


def test_affiliate_url_is_one_constant_and_labelled_in_docs():
    root = Path(__file__).parents[1]
    matches = [
        p.relative_to(root).as_posix()
        for p in (root / "src").rglob("*.py")
        if WAVESPEED_AFFILIATE_URL in p.read_text()
    ]
    assert matches == ["src/yap_studio/affiliate.py"]
    for path in (root / "docs").rglob("*.md"):
        for line in path.read_text().splitlines():
            if WAVESPEED_AFFILIATE_URL in line:
                assert "(affiliate link)" in line
    assert "(affiliate link): " + WAVESPEED_AFFILIATE_URL in wavespeed.SIGNUP_LINE


def test_wavespeed_signup_url_literals_and_docs():
    assert ELEVENLABS_AFFILIATE_URL == WAVESPEED_AFFILIATE_URL.replace(
        "/recommends/wavespeed", "/recommend/elevenlabs"
    )
    assert WAVESPEED_AFFILIATE_URL == "https://littlemight.com/recommends/wavespeed"
    assert wavespeed.SIGNUP_LINE == (
        "Sign up for WaveSpeed (affiliate link): https://littlemight.com/recommends/wavespeed"
    )
    root = Path(__file__).parents[1]
    for name in ("docs/install.md", "workspace/.claude/skills/edit-video/SKILL.md"):
        assert WAVESPEED_AFFILIATE_URL in (root / name).read_text()
    for folder in ("src", "docs", "workspace"):
        for path in (root / folder).rglob("*"):
            if path.is_file():
                assert (
                    b"https://littlemight.com/recommend/wavespeed" not in path.read_bytes()
                ), str(path)


def test_missing_cli_and_tool_contract():
    assert wavespeed.status() == "missing_cli"
    assert "wavespeed-pp-cli" in tools.ALLOWED_TOOLS
    error = DependencyError.missing_wavespeed_cli()
    assert error.suggested_action == wavespeed.INSTALL_LINE + "\n" + wavespeed.KEY_LINE
    with pytest.raises(DependencyError, match="WaveSpeed"):
        tools.run_tool("wavespeed-pp-cli", [], timeout_s=1)


@pytest.mark.parametrize(
    "reason,signup",
    [
        ("missing_cli", True),
        ("signed_out", True),
        ("offline", False),
        ("rate_limited", False),
        ("price_failed", False),
        ("unknown", False),
    ],
)
def test_unavailable_signup_only_for_missing_accounts(reason, signup):
    message = wavespeed.unavailable_message(reason)
    assert "\n" not in message
    assert "Generated pictures are unavailable:" in message
    assert "Carrying on with your own footage." in message
    if signup:
        assert message.endswith(wavespeed.SIGNUP_LINE)
    else:
        assert "Sign up" not in message and WAVESPEED_AFFILIATE_URL not in message


@pytest.mark.parametrize("code,state", [(0, "ready"), (4, "signed_out"), (5, "unknown")])
@pytest.mark.parametrize("bare", [False, True])
def test_status(fake_wavespeed, monkeypatch, code, state, bare):
    monkeypatch.setenv("FAKE_WAVESPEED_AUTH_EXIT", str(code))
    if bare:
        monkeypatch.setenv("FAKE_WAVESPEED_BARE", "1")
    assert wavespeed.status() == state
    assert calls(fake_wavespeed) == [["auth", "status", *wavespeed.FLAGS]]


@pytest.mark.parametrize(
    "code,stderr,reason",
    [
        (4, "signed out", "signed_out"),
        (7, "limited", "rate_limited"),
        (5, "HTTP 429", "rate_limited"),
        (5, "dial tcp: no such host", "offline"),
        (5, "TLS handshake timeout", "offline"),
        (5, "bad response", "price_failed"),
    ],
)
def test_estimate_failures(fake_wavespeed, monkeypatch, code, stderr, reason):
    monkeypatch.setenv("FAKE_WAVESPEED_PRICE_EXIT", str(code))
    monkeypatch.setenv("FAKE_WAVESPEED_PRICE_STDERR", stderr + "\nBearer synthetic-sensitive")
    with pytest.raises(wavespeed.PicturesUnavailable) as error:
        wavespeed.estimate("A park at dusk")
    assert error.value.reason == reason
    assert "synthetic-sensitive" not in str(error.value.to_dict())


@pytest.mark.parametrize("wrapped", [False, True])
def test_estimate_flags(fake_wavespeed, monkeypatch, wrapped):
    if wrapped:
        monkeypatch.setenv("FAKE_WAVESPEED_WRAPPED", "1")
    assert wavespeed.estimate("--help park") == 0.027
    argv = calls(fake_wavespeed)[0]
    assert argv[:3] == ["price", "--model-id", wavespeed.MODEL_ID]
    assert "--prompt=--help park" in argv and "size=" + wavespeed.SIZE in argv
    for flag in wavespeed.FLAGS:
        assert flag in argv
    assert "--agent" not in argv and "run" not in argv


@pytest.mark.parametrize("price", [-1, float("nan"), float("inf"), True, "0.027", None])
def test_bad_prices_are_not_quotes(monkeypatch, price):
    monkeypatch.setattr(
        tools,
        "run_tool",
        lambda *a, **kw: SimpleNamespace(stdout=json.dumps({"data": {"unit_price": price}})),
    )
    with pytest.raises(wavespeed.PicturesUnavailable, match="unavailable"):
        wavespeed.estimate("A park")


def test_dry_run_is_redacted_and_writes_nothing(tmp_path, fake_wavespeed):
    target = tmp_path / "not-created"
    preview = wavespeed.generate("A quiet park", target, dry_run=True)
    assert "POST https://" in preview and '"prompt": "A quiet park"' in preview
    assert "Authorization" not in preview and "Bearer" not in preview
    assert not target.exists()
    argv = calls(fake_wavespeed)[0]
    assert "--dry-run" in argv and "--no-learn" in argv
    assert any(a.startswith("--download=") for a in argv) and "--download" not in argv


def test_submitted_failure_keeps_id_without_retry(tmp_path, fake_wavespeed, monkeypatch):
    monkeypatch.setenv("FAKE_WAVESPEED_LOST", "1")
    with pytest.raises(wavespeed.GenerationError) as error:
        wavespeed.generate("A quiet park", tmp_path, dry_run=False)
    assert wavespeed.prediction_id(error.value.prediction_id)
    assert error.value.price == 0.027
    assert len(calls(fake_wavespeed)) == 1


@pytest.mark.parametrize("identifier", ["../bad", "a" * 81, "", None, 42])
def test_invalid_billing_ids_make_no_call(fake_wavespeed, identifier):
    assert wavespeed.charged(identifier) is None
    assert not calls(fake_wavespeed)


def test_timeout_preserves_prediction_stdout(tmp_path, fake_wavespeed, monkeypatch):
    payload = {"prediction": {"data": {"id": "known123", "status": "completed"}}}

    def expired(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=json.dumps(payload).encode()
        )

    monkeypatch.setattr(process.subprocess, "run", expired)
    with pytest.raises(wavespeed.GenerationError) as error:
        wavespeed.generate("A park", tmp_path, dry_run=False)
    assert error.value.prediction_id == "known123"
