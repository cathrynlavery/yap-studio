"""Offline Pexels parsing, fit, transport, and dependency contracts."""

from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import replace
from pathlib import Path

import pytest

import yap_studio
from yap_studio import tools
from yap_studio.errors import DependencyError, ValidationError

FIXTURES = Path(__file__).parent / "fixtures/stock"
ROOMY_S = 60  # Load cannot run this out; the deadline itself uses a fake clock.


def calls(log):
    return (
        [json.loads(line)["argv"] for line in log.read_text().splitlines()] if log.exists() else []
    )


def test_missing_dependency_and_no_payloads():
    from yap_studio.stock.pexels import PexelsProvider

    assert PexelsProvider().status() == "missing_cli"
    error = DependencyError.missing_pexels_cli()
    assert "install pexels --cli-only" in error.suggested_action
    assert "https://www.pexels.com/api/" in error.suggested_action
    with pytest.raises(DependencyError, match="Pexels"):
        tools.run_tool("pexels-pp-cli", [], timeout_s=1)
    for option in ({"input_bytes": b"no"}, {"stdout_path": "no"}):
        with pytest.raises(ValueError, match="only supported for elevenlabs"):
            tools.run_tool("pexels-pp-cli", [], timeout_s=1, **option)


@pytest.mark.parametrize(("code", "state"), [(0, "ready"), (4, "signed_out"), (1, "unknown")])
def test_status(fake_pexels, monkeypatch, code, state):
    from yap_studio.stock.pexels import PexelsProvider

    monkeypatch.setenv("FAKE_PEXELS_AUTH_EXIT", str(code))
    assert PexelsProvider().status() == state


@pytest.mark.parametrize(
    ("code", "stderr", "reason"),
    [
        (4, "no saved key", "signed_out"),
        (1, "HTTP 429", "rate_limited"),
        (1, "rate limit reached", "rate_limited"),
        (1, "dial tcp: no such host", "offline"),
        (1, "TLS handshake timeout", "offline"),
        (1, "bad response", "search_failed"),
    ],
)
def test_failures_redact(fake_pexels, monkeypatch, code, stderr, reason):
    from yap_studio.stock.pexels import PexelsProvider
    from yap_studio.stock.provider import StockUnavailable

    monkeypatch.setenv("FAKE_PEXELS_SEARCH_EXIT", str(code))
    monkeypatch.setenv(
        "FAKE_PEXELS_SEARCH_STDERR",
        stderr + "\nAuthorization: private\napi key private\napi_key private\ntoken private",
    )
    with pytest.raises(StockUnavailable) as error:
        PexelsProvider().search("dog park", orientation="portrait", size="medium", per_page=15)
    assert error.value.reason == reason
    assert "private" not in str(error.value)
    assert "private" not in str(error.value.to_dict())


def test_parsing_fit_and_flags(fake_pexels):
    from yap_studio.stock.pexels import PexelsProvider
    from yap_studio.stock.provider import best_file, crop_width, in_point, tier

    provider = PexelsProvider()
    clips = provider.search("dog park", orientation=None, size="large", per_page=15)
    by_id = {clip.clip_id: clip for clip in clips}
    assert crop_width(1080, 1920) == 1080
    assert crop_width(1920, 1080) == 607.5
    assert crop_width(3840, 2160) == 1215
    assert best_file(by_id["101"]).file_id == "1010"
    assert tier(by_id["101"]) == 0 and tier(by_id["102"]) == 1
    assert best_file(by_id["103"]) is None
    assert best_file(by_id["105"]) is None
    assert by_id["101"].preview_url.endswith("/2.jpg")
    assert provider.credit(by_id["101"]) == "Video by Alex <b>& on Pexels"
    assert provider.details("101") == by_id["101"]
    argv = calls(fake_pexels)[0]
    for flag in (
        "--json",
        "--no-input",
        "--no-color",
        "--data-source",
        "live",
        "--timeout",
        "30s",
        "--page",
        "1",
    ):
        assert flag in argv
    assert "--dry-run" not in argv
    assert in_point(12, 3) == 1 and in_point(3.25, 3) == 0.25
    assert in_point(3, 3) == 0 and in_point(1, 3) == 0
    file = best_file(by_id["101"])
    clip = replace(
        by_id["101"],
        files=(
            replace(file, file_id="20", fps=24),
            replace(file, file_id="10", fps=30),
            replace(file, file_id="2", fps=30),
            replace(file, file_id="1", width=2160, height=3840),
        ),
    )
    assert best_file(clip).file_id == "2"


@pytest.mark.parametrize("shape", ["envelope", "page", "list"])
def test_defensive_parsing(fake_pexels, monkeypatch, shape):
    from yap_studio.stock.pexels import PexelsProvider

    payload = json.loads((FIXTURES / "portrait.json").read_text())
    videos = payload["results"]["videos"]
    videos.extend([None, {}, {"id": 9}, {**videos[0], "duration": "bad"}])
    videos[0]["video_files"].extend([None, {}, {"file_type": "video/mp4", "link": "http://bad"}])
    if shape != "envelope":
        payload = payload["results"] if shape == "page" else videos
    monkeypatch.setattr(
        tools, "run_tool", lambda *a, **kw: type("Result", (), {"stdout": json.dumps(payload)})()
    )
    assert (
        len(PexelsProvider().search("dog", orientation="portrait", size="medium", per_page=15)) == 4
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://videos.pexels.com/a",
        "https://evil.com/a",
        "https://videos.pexels.com.evil.com/a",
        "https://user@videos.pexels.com/a",
    ],
)
def test_fetch_refuses_hosts(tmp_path, url):
    from yap_studio.stock.fetch import fetch

    with pytest.raises(ValidationError, match="url"):
        fetch(url, tmp_path / "a.mp4", max_bytes=10, timeout_s=ROOMY_S)
    assert not list(tmp_path.iterdir())


def test_fetch_stream_limits_redirects_and_offline(tmp_path, monkeypatch):
    import yap_studio.stock.fetch as transport
    from yap_studio.stock.provider import StockUnavailable

    class Response(io.BytesIO):
        headers = {}

    class Opener:
        def open(self, request, timeout):
            assert timeout == ROOMY_S
            assert request.get_header("User-agent") == f"yap-studio/{yap_studio.__version__}"
            return Response(b"123456")

    monkeypatch.setattr(transport.urllib.request, "build_opener", lambda *a: Opener())
    target = tmp_path / "a.mp4"
    target.write_bytes(b"old")
    with pytest.raises(ValidationError, match="max_bytes"):
        transport.fetch("https://videos.pexels.com/a", target, max_bytes=5, timeout_s=ROOMY_S)
    assert target.read_bytes() == b"old" and not target.with_suffix(".mp4.part").exists()
    transport.fetch("https://videos.pexels.com/a", target, max_bytes=6, timeout_s=ROOMY_S)
    assert target.read_bytes() == b"123456"
    redirect = transport.CheckedRedirect()
    request = transport.urllib.request.Request("https://videos.pexels.com/a")
    with pytest.raises(ValidationError, match="url"):
        redirect.redirect_request(request, None, 302, "", {}, "https://evil.com/a")
    for _ in range(3):
        request = redirect.redirect_request(
            request, None, 302, "", {}, "https://videos.pexels.com/b"
        )
    with pytest.raises(ValidationError, match="redirect"):
        redirect.redirect_request(request, None, 302, "", {}, "https://videos.pexels.com/c")

    def offline(*a):
        raise urllib.error.URLError(TimeoutError("timeout"))

    monkeypatch.setattr(transport.urllib.request, "build_opener", offline)
    with pytest.raises(StockUnavailable, match="internet"):
        transport.fetch("https://videos.pexels.com/a", target, max_bytes=6, timeout_s=ROOMY_S)


@pytest.mark.parametrize("pictures", [17, {}, "bad", [None, {"nr": 1}, {"picture": "bad"}]])
def test_malformed_optional_pictures_keep_the_video(fake_pexels, monkeypatch, pictures):
    from yap_studio.stock.pexels import PexelsProvider

    video = json.loads((FIXTURES / "101.json").read_text())["results"]
    video["video_pictures"] = pictures
    monkeypatch.setattr(
        tools, "run_tool", lambda *a, **kw: type("Result", (), {"stdout": json.dumps(video)})()
    )
    clip = PexelsProvider().details("101")
    assert clip.preview_url == video["image"] and len(clip.files) == 1


def test_missing_search_cli_and_bare_details(monkeypatch):
    from yap_studio.stock.pexels import PexelsProvider
    from yap_studio.stock.provider import StockUnavailable

    with pytest.raises(StockUnavailable) as error:
        PexelsProvider().search("dog park", orientation="portrait", size="medium", per_page=15)
    assert error.value.reason == "missing_cli" and len(error.value.setup) == 2
    video = json.loads((FIXTURES / "101.json").read_text())["results"]
    monkeypatch.setattr(
        tools, "run_tool", lambda *a, **kw: type("Result", (), {"stdout": json.dumps(video)})()
    )
    assert PexelsProvider().details("101").clip_id == "101"


def test_fetch_socket_error(tmp_path, monkeypatch):
    import socket

    import yap_studio.stock.fetch as transport
    from yap_studio.stock.provider import StockUnavailable

    class DNSFailure:
        def open(self, request, timeout):
            assert timeout == ROOMY_S
            raise socket.gaierror("synthetic DNS failure")

    monkeypatch.setattr(transport.urllib.request, "build_opener", lambda *a: DNSFailure())
    with pytest.raises(StockUnavailable) as error:
        transport.fetch(
            "https://videos.pexels.com/a", tmp_path / "a.mp4", max_bytes=10, timeout_s=ROOMY_S
        )
    assert error.value.reason == "offline"


@pytest.mark.parametrize(("step", "expires"), [(2, True), (0.25, False)])
def test_fetch_elapsed_limit(tmp_path, monkeypatch, step, expires):
    import yap_studio.stock.fetch as transport
    from yap_studio.stock.provider import StockUnavailable

    class SlowResponse(io.BytesIO):
        headers = {}

    class SlowOpener:
        def open(self, request, timeout):
            assert timeout == 1
            return SlowResponse(b"1234")

    monkeypatch.setattr(transport.urllib.request, "build_opener", lambda *a: SlowOpener())
    clock = [0.0]

    def elapsed():
        clock[0] += step
        return clock[0]

    target = tmp_path / "a.mp4"
    target.write_bytes(b"old")
    if expires:
        with pytest.raises(StockUnavailable) as error:
            transport.fetch(
                "https://videos.pexels.com/a", target, max_bytes=10, timeout_s=1, clock=elapsed
            )
        assert error.value.reason == "offline"
        assert isinstance(error.value.__cause__, TimeoutError)
        assert target.read_bytes() == b"old"
    else:
        transport.fetch(
            "https://videos.pexels.com/a", target, max_bytes=10, timeout_s=1, clock=elapsed
        )
        assert target.read_bytes() == b"1234"
    assert not (tmp_path / "a.mp4.part").exists()


def test_dash_query_is_one_option_argument(fake_pexels):
    from yap_studio.stock.pexels import PexelsProvider

    query = "--help quiet park"
    PexelsProvider().search(query, orientation="portrait", size="medium", per_page=15)
    argv = calls(fake_pexels)[0]
    assert f"--query={query}" in argv
    assert "--query" not in argv and query not in argv


@pytest.mark.parametrize("status", [404, 403, 503])
def test_fetch_http_error_is_download_failed(tmp_path, monkeypatch, status):
    import yap_studio.stock.fetch as transport
    from yap_studio.stock.provider import StockUnavailable

    class Opener:
        def open(self, request, timeout):
            assert timeout == ROOMY_S
            raise urllib.error.HTTPError(request.full_url, status, "synthetic error", {}, None)

    monkeypatch.setattr(transport.urllib.request, "build_opener", lambda *a: Opener())
    target = tmp_path / "clip.mp4"
    with pytest.raises(StockUnavailable) as error:
        transport.fetch(
            "https://videos.pexels.com/clip.mp4", target, max_bytes=10, timeout_s=ROOMY_S
        )
    assert error.value.reason == "download_failed"
    assert error.value.code == "download_failed"
    assert f"Pexels could not send the file (HTTP {status})" in error.value.message
    assert "no internet connection" not in error.value.message
    assert not target.exists() and not target.with_name("clip.mp4.part").exists()


@pytest.mark.parametrize("length", ["not a length", "", "1.5"])
@pytest.mark.parametrize("limit", [5, 6])
def test_fetch_malformed_content_length_is_unknown(tmp_path, monkeypatch, length, limit):
    import yap_studio.stock.fetch as transport

    class Response(io.BytesIO):
        headers = {"Content-Length": length}

    class Opener:
        def open(self, request, timeout):
            assert timeout == ROOMY_S
            return Response(b"123456")

    monkeypatch.setattr(transport.urllib.request, "build_opener", lambda *a: Opener())
    target = tmp_path / "clip.mp4"
    if limit == 5:
        with pytest.raises(ValidationError, match="max_bytes"):
            transport.fetch(
                "https://videos.pexels.com/clip.mp4", target, max_bytes=limit, timeout_s=ROOMY_S
            )
        assert not target.exists()
    else:
        transport.fetch(
            "https://videos.pexels.com/clip.mp4", target, max_bytes=limit, timeout_s=ROOMY_S
        )
        assert target.read_bytes() == b"123456"
    assert not target.with_name("clip.mp4.part").exists()


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "http://www.pexels.com/@alex/",
        "https://",
        "/alex",
        7,
        "https://user@www.pexels.com/alex",
        "https://[invalid",
    ],
)
def test_author_url_falls_back_to_clip_page(monkeypatch, url):
    from yap_studio.stock.metadata import validate_stock
    from yap_studio.stock.pexels import PexelsProvider

    video = json.loads((FIXTURES / "101.json").read_text())["results"]
    if url is None:
        video["user"].pop("url")
    else:
        video["user"]["url"] = url
    monkeypatch.setattr(
        tools, "run_tool", lambda *a, **kw: type("Result", (), {"stdout": json.dumps(video)})()
    )
    clip = PexelsProvider().details("101")
    assert clip.author_url == clip.page_url
    validate_stock(
        {
            "provider": clip.provider,
            "clip_id": clip.clip_id,
            "page_url": clip.page_url,
            "author": clip.author,
            "author_url": clip.author_url,
            "license": clip.license,
            "license_url": clip.license_url,
            "file_url": clip.files[0].url,
            "downloaded_at": "2026-10-18T12:00:00Z",
        }
    )


def test_pexels_guard_preserves_other_apps_config(monkeypatch, tmp_path, _owner_folders):
    import os

    from conftest import _no_real_pexels

    expected = str(tmp_path / "other-app-config")
    monkeypatch.setenv("XDG_CONFIG_HOME", expected)
    with pytest.MonkeyPatch.context() as isolated:
        guard = _no_real_pexels.__wrapped__(isolated, tmp_path, _owner_folders)
        try:
            next(guard)
            assert os.environ["XDG_CONFIG_HOME"] == expected
            assert os.environ["PEXELS_HOME"] == str(tmp_path / "pexels")
            assert "PEXELS_API_KEY" not in os.environ
        finally:
            guard.close()
