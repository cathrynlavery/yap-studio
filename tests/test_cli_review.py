"""Review command tests."""

from __future__ import annotations

import errno
import http.client
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from yap_studio import receipt
from yap_studio.cli import main
from yap_studio.review import server as server_module
from yap_studio.review.launch import version_video
from yap_studio.review.media import ReviewMedia
from yap_studio.review.server import ReviewServer
from yap_studio.review.store import ReviewStore

ROOT = Path(__file__).parents[1]


def test_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    assert main(["review", str(tmp_path / "missing.mp4"), "--no-open"]) == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err and "fix:" in captured.err


def test_review_argument_error_json_and_plain_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["review", "x.mp4", "--port", "nope", "--json"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    error = json.loads(output.err)["error"]
    assert error["code"] == "invalid_argument"
    assert "argument --port" in error["message"]

    assert main(["review", "x.mp4", "--port", "nope"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "usage: yap-studio review" in output.err


@pytest.mark.parametrize("arguments", [
    [], ["--stop", "--port", "0"], ["missing.mp4"],
])
def test_review_json_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    assert main(["review", *arguments, "--json"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    error = json.loads(output.err)["error"]
    assert error["code"] and error["message"]


def test_versions_one_file_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    video = tmp_path / "only.mp4"
    video.write_bytes(b"one render")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    assert main(["review", "--versions", str(video), "--no-open"]) == 2
    error = capsys.readouterr().err
    assert "error:" in error and "fix:" in error
    assert "--versions needs at least two files, oldest first, or a project folder" in error


def test_versions_identical_files_name_both_supplied_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    values = ["older/out.mp4", "./newer/out.mp4"]
    for value in values:
        video = Path(value)
        video.parent.mkdir()
        video.write_bytes(b"identical render")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        "yap_studio.review.launch.probe_media",
        lambda path: ReviewMedia(Path(path), Path(path).name, 2.0, None, False, "h264", 64, 36),
    )
    assert main(["review", "--versions", *values, "--no-open"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "Bad value for media: older/out.mp4 and ./newer/out.mp4 have identical contents; "
        "each version must differ"
    ) in captured.err
    assert not (tmp_path / "data" / "series").exists()


def test_versions_review_prints_one_link_and_passes_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    videos = [tmp_path / "first.mp4", tmp_path / "second.mp4"]
    for number, video in enumerate(videos):
        video.write_bytes(f"render {number}".encode())
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        "yap_studio.review.launch.probe_media",
        lambda path: ReviewMedia(Path(path), Path(path).name, 2.0, None, False, "h264", 64, 36),
    )
    seen = []

    def serving(server: ReviewServer) -> None:
        assert len(server.videos) == 1 and server.list_url is None
        assert [item.media.name for item in server.series.versions] == ["first.mp4", "second.mp4"]
        seen.append(server.url)
        raise KeyboardInterrupt

    monkeypatch.setattr(ReviewServer, "serve_forever", serving)
    assert main(["review", "--versions", *(str(path) for path in videos),
                 "--no-open", "--json"]) == 0
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    document = json.loads(lines[0])
    assert document["list"] is None
    assert document["videos"] == [{"name": "second.mp4", "sha256": receipt.sha256_file(videos[1]),
                                    "url": seen[0]}]
    assert captured.err == ""


@pytest.mark.parametrize("live", [False, True])
def test_durable_series_token_port_restart_and_busy_fallback(
    tmp_path: Path, monkeypatch, live,
) -> None:
    files = [tmp_path / "first.mp4", tmp_path / "second.mp4"]
    for index, file in enumerate(files):
        file.write_bytes(f"render {index}".encode())
    monkeypatch.setattr("yap_studio.review.launch.probe_media", lambda path:
                        ReviewMedia(path, path.name, 2.0, None, False, "h264", 64, 36))
    bindings = set()

    class Binding:
        def __init__(self, address, _handler):
            requested = address[1]
            if requested in bindings:
                raise OSError(errno.EADDRINUSE, "busy")
            self.server_port = requested or next(port for port in range(20000, 20010)
                                                if port not in bindings)
            bindings.add(self.server_port)

        def server_close(self):
            bindings.remove(self.server_port)

    if not live:
        monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer", Binding)
    root = tmp_path / "data"
    values = [str(file) for file in files]

    def start():
        store = ReviewStore(root)
        video = version_video(values, store, lambda _: files)
        return ReviewServer(video.media, video.sha256, None, store, token=video.token,
                            port=0, series=video.series)

    first = start()
    first_url, first_port = first.url, first.port
    first.server_close()
    second = start()
    try:
        assert second.url == first_url and second.port == first_port
        third = start()
        try:
            assert third.token == second.token and third.port != second.port
            assert ReviewStore(root).series_link(third.series.id)["port"] == third.port
        finally:
            third.server_close()
    finally:
        second.server_close()


def test_explicit_port_wins_over_saved_port(tmp_path: Path, monkeypatch, capsys) -> None:
    files = [tmp_path / "first.mp4", tmp_path / "second.mp4"]
    for index, file in enumerate(files):
        file.write_bytes(f"render {index}".encode())
    store = ReviewStore(tmp_path / "data")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(store.root))
    monkeypatch.setattr("yap_studio.review.launch.probe_media", lambda path:
                        ReviewMedia(path, path.name, 2.0, None, False, "h264", 64, 36))
    video = version_video([str(file) for file in files], store, lambda _: files)
    store.save_link_port(video.series.id, 12345)
    addresses = []

    def bind(address, _handler):
        addresses.append(address)
        return SimpleNamespace(server_port=address[1], server_close=lambda: None,
                               serve_forever=lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer", bind)
    assert main(["review", "--versions", *(str(file) for file in files), "--no-open",
                 "--port", "23456", "--json"]) == 0
    assert addresses == [("127.0.0.1", 23456)]
    assert ":23456/" in json.loads(capsys.readouterr().out)["videos"][0]["url"]
    assert store.series_link(video.series.id)["port"] == 23456


def test_port_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    assert main(["review", "sample.mp4", "--port", "65536", "--no-open"]) == 2
    assert "port must be" in capsys.readouterr().err


@pytest.mark.parametrize(
    "value",
    [
        "", " ", "review example.net", "https://review.example.net", "review.example.net/path",
        "*.example.net", "review.example.net:0", "review.example.net:65536",
        "review.example.net:abc", "review.example.net:", "user@review.example.net", "[::1]",
    ],
)
def test_invalid_allow_host(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["review", "sample.mp4", "--allow-host", value, "--no-open"]) == 2
    error = capsys.readouterr().err
    assert "--allow-host" in error
    assert "hostname or IPv4" in error
    assert "1 to 65535" in error


@pytest.mark.parametrize(
    "value",
    [
        "", "review.example.net", "ftp://review.example.net", "https://",
        "https://review.example.net/", "https://review.example.net/path",
        "https://review.example.net?x=1", "https://review.example.net?",
        "https://review.example.net#fragment", "https://review.example.net#",
        "https://user@review.example.net", "https://review.example.net:0",
        "https://review.example.net:65536", "https://review.example.net:abc",
        "https://review.example.net:", "https://*.example.net", "http://[::1]",
        " https://review.example.net", "https://review.example.net ",
        "https://review.example.net\n", "https://review.\texample.net",
    ],
)
def test_invalid_public_base(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["review", "sample.mp4", "--public-base", value, "--no-open"]) == 2
    error = capsys.readouterr().err
    assert "--public-base" in error
    assert "http:// or https:// origin" in error
    assert "no path, query or fragment" in error


@pytest.mark.parametrize(
    "public_base",
    [None, "https://box.example.net:8443", "https://box.example.net", "http://192.0.2.10:8080"],
)
def test_review_proxy_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    public_base: str | None,
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"synthetic video bytes")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        "yap_studio.review.media.probe_media",
        lambda path: ReviewMedia(Path(path), "sample.mp4", None, None, False, "h264", 64, 36),
    )
    opened: list[str] = []
    monkeypatch.setattr("yap_studio.cli.webbrowser.open", opened.append)
    serve_forever = ReviewServer.serve_forever
    local_links: list[str] = []

    def get(server: ReviewServer, host: str, route: str) -> int:
        connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        try:
            connection.request(
                "GET", route + "?" + urlsplit(server.url).query, headers={"Host": host}
            )
            response = connection.getresponse()
            response.read()
            return response.status
        finally:
            connection.close()

    def check_hosts(server: ReviewServer) -> None:
        local_links.append(server.url)
        assert server._http.server_address[0] == "127.0.0.1"
        thread = threading.Thread(target=serve_forever, args=(server,), daemon=True)
        thread.start()
        try:
            hosts = ["review.example.net:443", "other.example.net:8080"]
            if public_base is not None:
                hosts.append(urlsplit(public_base).netloc.upper())
            for host in hosts:
                for route in ("/", "/api/review", "/media"):
                    assert get(server, host, route) == 200
            if public_base == "https://box.example.net:8443":
                assert get(server, "box.example.net:8444", "/") == 403
        finally:
            server.shutdown()
            thread.join(timeout=5)
        raise KeyboardInterrupt

    monkeypatch.setattr(ReviewServer, "serve_forever", check_hosts)
    argv = [
        "review", str(video), "--allow-host", "review.example.net",
        "--allow-host", "other.example.net:8080",
    ]
    if public_base is not None:
        argv.extend(["--public-base", public_base])
    assert main(argv) == 0
    local = urlsplit(local_links[0])
    expected = (
        f"{public_base}{local.path}?{local.query}" if public_base is not None else local_links[0]
    )
    assert capsys.readouterr().out == (
        f"Reviewing sample.mp4 at {expected}\nPress Ctrl-C to stop.\n"
    )
    assert opened == [expected]


def test_review_directory_expands_in_name_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "videos"
    folder.mkdir()
    names = ["e.webm", "b.MOV", "d.mkv", "a.mp4", "c.m4v"]
    for name in names:
        (folder / name).write_bytes(f"synthetic {name}".encode())
    (folder / "notes.txt").write_text("ignored", encoding="utf-8")
    nested = folder / "nested.mp4"
    nested.mkdir()
    (nested / "hidden.mp4").write_bytes(b"ignored nested video")
    before = tmp_path / "before.mp4"
    before.write_bytes(b"synthetic first argument")
    after = tmp_path / "after.mp4"
    after.write_bytes(b"synthetic last argument")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    probed: list[str] = []

    def probe(path: Path) -> ReviewMedia:
        probed.append(path.name)
        return ReviewMedia(path, path.name, 1.0, None, False, "h264", 64, 36)

    monkeypatch.setattr("yap_studio.review.media.probe_media", probe)
    monkeypatch.setattr(
        "yap_studio.review.server._QuietThreadingHTTPServer",
        lambda *_args: SimpleNamespace(server_port=12345, server_close=lambda: None),
    )
    monkeypatch.setattr(ReviewServer, "serve_forever", lambda _self: None)
    assert main(["review", str(before), str(folder), str(after), "--no-open"]) == 0
    expected = [before.name, *sorted(names), after.name]
    assert probed == expected
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Review list at http://127.0.0.1:12345/?t=")
    assert lines[-1] == "Press Ctrl-C to stop."
    for number, (name, line) in enumerate(zip(expected, lines[1:-1], strict=True), 1):
        assert line.startswith(f"{name}  http://127.0.0.1:12345/v/{number}/?t=")


@pytest.mark.parametrize("duplicate_kind", ["same_path", "same_contents", "directory_overlap"])
def test_duplicate_video_refused_before_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    duplicate_kind: str,
) -> None:
    first = tmp_path / "first.mp4"
    first.write_bytes(b"synthetic matching contents")
    other = tmp_path / "other.mp4"
    other.write_bytes(first.read_bytes())
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        "yap_studio.review.media.probe_media",
        lambda path: ReviewMedia(path, path.name, 1.0, None, False, "h264", 64, 36),
    )
    monkeypatch.setattr(
        ReviewServer, "__init__", lambda *_args, **_kwargs: pytest.fail("duplicate bound a port")
    )
    duplicate = {
        "same_path": first, "same_contents": other, "directory_overlap": tmp_path,
    }[duplicate_kind]
    assert main(["review", str(first), str(duplicate), "--no-open"]) == 2
    captured = capsys.readouterr()
    assert "duplicate video" in captured.err and "fix:" in captured.err
    assert captured.out == ""


def test_empty_review_directory_is_an_input_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert main(["review", str(tmp_path), "--no-open"]) == 2
    assert "no video files" in capsys.readouterr().err


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("public_base", [None, "https://review.example.net:8443"])
def test_review_json_prints_working_links_before_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    count: int, public_base: str | None,
) -> None:
    paths = [tmp_path / f"sample-{number}.mp4" for number in range(count)]
    for number, path in enumerate(paths):
        path.write_bytes(f"synthetic video {number}".encode())
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        "yap_studio.review.media.probe_media",
        lambda path: ReviewMedia(path, path.name, 1.0, None, False, "h264", 64, 36),
    )
    opened: list[str] = []
    monkeypatch.setattr("yap_studio.cli.webbrowser.open", opened.append)
    serve_forever = ReviewServer.serve_forever
    documents: list[dict] = []

    def check_links(server: ReviewServer) -> None:
        output = capsys.readouterr().out.splitlines()
        assert len(output) == 1
        document = json.loads(output[0])
        documents.append(document)
        assert set(document) == {"list", "videos"}
        assert len(document["videos"]) == count
        assert (document["list"] is None) is (count == 1)
        assert [item["name"] for item in document["videos"]] == [path.name for path in paths]
        thread = threading.Thread(target=serve_forever, args=(server,), daemon=True)
        thread.start()
        try:
            for number, item in enumerate(document["videos"], 1):
                assert set(item) == ({"name", "sha256", "url", "local_url"}
                                     if public_base is not None else {"name", "sha256", "url"})
                if public_base is not None:
                    assert urlsplit(item["local_url"]).netloc == f"127.0.0.1:{server.port}"
                assert item["sha256"] == receipt.sha256_file(paths[number - 1])
                parsed = urlsplit(item["url"])
                assert parsed.path == ("/" if count == 1 else f"/v/{number}/")
                if public_base is not None:
                    assert item["url"].startswith(public_base + "/")
                else:
                    assert parsed.netloc == f"127.0.0.1:{server.port}"
                for suffix in ("", "api/review", "media"):
                    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
                    try:
                        connection.request(
                            "GET", parsed.path + suffix + "?" + parsed.query,
                            headers={"Host": parsed.netloc},
                        )
                        response = connection.getresponse()
                        body = response.read()
                        assert response.status == 200
                        if suffix == "api/review":
                            assert json.loads(body)["media"]["sha256"] == item["sha256"]
                    finally:
                        connection.close()
            if count > 1:
                parsed = urlsplit(document["list"])
                if public_base is not None:
                    assert document["list"].startswith(public_base + "/")
                connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
                try:
                    connection.request(
                        "GET", parsed.path + "?" + parsed.query, headers={"Host": parsed.netloc}
                    )
                    response = connection.getresponse()
                    body = response.read()
                    assert response.status == 200
                    assert all(item["name"].encode() in body for item in document["videos"])
                finally:
                    connection.close()
        finally:
            server.shutdown()
            thread.join(timeout=5)
        raise KeyboardInterrupt

    monkeypatch.setattr(ReviewServer, "serve_forever", check_links)
    argv = ["review", *(str(path) for path in paths), "--json"]
    if public_base is not None:
        argv.extend(["--public-base", public_base])
    assert main(argv) == 0
    document = documents[0]
    assert opened == [document["list"] or document["videos"][0]["url"]]
    assert capsys.readouterr().out == ""


def test_unwritable_data_dir_is_clean_command_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    if os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0):
        pytest.skip("chmod permissions are not enforced for this user")
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "yap_studio.review.media.probe_media",
        lambda path: ReviewMedia(Path(path), "sample.mp4", None, None, False, "h264", 64, 36),
    )
    parent = tmp_path / "read-only"
    parent.mkdir()
    parent.chmod(0o500)
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(parent / "data"))
    try:
        assert main(["review", str(video), "--no-open"]) == 2
        captured = capsys.readouterr()
        assert "error: Cannot write review data" in captured.err
        assert "fix: Set YAP_STUDIO_DATA_DIR to a writable folder." in captured.err
        assert "Traceback" not in captured.err
    finally:
        parent.chmod(0o700)


@pytest.mark.parametrize("file_kind", ["review", "settings"])
def test_corrupt_review_data_stops_before_server_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    file_kind: str,
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "yap_studio.review.media.probe_media",
        lambda path: ReviewMedia(Path(path), "sample.mp4", None, None, False, "h264", 64, 36),
    )
    root = tmp_path / "data"
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(root))
    path = (
        root / "reviews" / f"{receipt.sha256_file(video)}.json"
        if file_kind == "review"
        else root / "settings.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(
        ReviewServer, "__init__", lambda *_args, **_kwargs: pytest.fail("server started")
    )

    assert main(["review", str(video), "--no-open"]) == 2
    captured = capsys.readouterr()
    assert path.name in captured.err
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    assert "Reviewing" not in captured.out


def test_review_interrupt_and_reviewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    from fractions import Fraction

    from yap_studio.review import media as media_module
    from yap_studio.review.media import ReviewMedia

    monkeypatch.setattr(
        media_module,
        "probe_media",
        lambda path: ReviewMedia(Path(path), "sample.mp4", 2.0, Fraction(30), True, "h264", 64, 36),
    )
    monkeypatch.setattr(
        ReviewServer, "serve_forever", lambda _self: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    assert main(["review", str(video), "--reviewer", "Alex Reviewer", "--no-open"]) == 0
    output = capsys.readouterr().out
    assert "Reviewing sample.mp4 at http://127.0.0.1:" in output
    assert "Press Ctrl-C to stop." in output
    assert ReviewStore(tmp_path / "data").reviewer() == "Alex Reviewer"


def test_probe_change_stops_before_server_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"first")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))

    def changing_probe(path: Path) -> ReviewMedia:
        video.write_bytes(b"new video contents")
        return ReviewMedia(Path(path), "sample.mp4", None, None, False, "h264", 64, 36)

    monkeypatch.setattr("yap_studio.review.media.probe_media", changing_probe)
    monkeypatch.setattr(
        ReviewServer, "__init__", lambda *_args, **_kwargs: pytest.fail("server started")
    )
    assert main(["review", str(video), "--no-open"]) == 2
    assert "the file changed while it was being opened; try again" in capsys.readouterr().err


@pytest.mark.e2e
def test_review_subprocess(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")
    video = tmp_path / "sample.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x36:rate=30000/1001",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=True,
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["YAP_STUDIO_DATA_DIR"] = str(tmp_path / "data")
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "yap_studio",
            "review",
            str(video),
            "--no-open",
            "--port",
            "0",
        ],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_lines: queue.Queue[str | None] = queue.Queue()
    stderr_lines: list[str] = []

    def read_stdout() -> None:
        for line in process.stdout:
            stdout_lines.put(line)
        stdout_lines.put(None)

    def read_stderr() -> None:
        stderr_lines.extend(process.stderr)

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    deadline = time.monotonic() + 120
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail(
                    f"review command did not print its URL; stderr: {''.join(stderr_lines)}"
                )
            try:
                line = stdout_lines.get(timeout=remaining)
            except queue.Empty:
                pytest.fail(
                    f"review command did not print its URL; stderr: {''.join(stderr_lines)}"
                )
            if line is None:
                pytest.fail(
                    "review command exited before printing its URL; "
                    f"stderr: {''.join(stderr_lines)}"
                )
            line = line.strip()
            if line.startswith("Reviewing sample.mp4 at http://"):
                break
        url = urlsplit(line.split(" at ", 1)[1])
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        connection.request(
            "GET", "/api/review" + "?" + url.query, headers={"Host": f"127.0.0.1:{url.port}"}
        )
        response = connection.getresponse()
        assert response.status == 200
        payload = json.loads(response.read())
        connection.close()
        assert payload["media"]["name"] == "sample.mp4"
        assert payload["media"]["cfr"] is True
        assert payload["comments"] == []
        process.send_signal(signal.SIGINT)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail(f"review command did not exit in time; stderr: {''.join(stderr_lines)}")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pytest.fail(f"review command did not exit in time; stderr: {''.join(stderr_lines)}")
        assert returncode == 0, (
            f"review command exited {returncode}; stderr: {''.join(stderr_lines)}"
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
