"""Privacy copy and phone transport labels, using isolated synthetic projects."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_feedback_batch import feedback_project as feedback_project
from test_review_phone import binds as binds
from test_review_phone import video as video
from test_review_server import _in_process_request
from test_review_server import removed_server as removed_server

from yap_studio import background, cli
from yap_studio.gallery import detach as style_detach
from yap_studio.homepage import detach as home_detach
from yap_studio.review import detach as review_detach
from yap_studio.review import phone

ROOT = Path(__file__).resolve().parents[1]
DISCLOSURE = (
    "Your footage, renders and review files stay on your computer. "
    "The transcript, your notes and the project files Claude reads go to Claude (Anthropic) "
    "as part of your Claude Code session. ElevenLabs, WaveSpeed and Pexels get only what "
    "they need, and only if you turn them on."
)
TRANSPORTS = [
    ("tailscale", "Encrypted (Tailscale)"),
    ("wifi", "Not encrypted (same Wi-Fi)"),
]


@pytest.mark.parametrize("name", [
    "README.md", "docs/install.md", "workspace/Start here.html",
    "src/yap_studio/homepage/static/home.js",
])
def test_entry_points_disclose_where_data_goes(name):
    text = (ROOT / name).read_text()
    if name.endswith("home.js"):
        text = text.split("function renderWelcome()", 1)[1].split("function stepHeader", 1)[0]
    assert DISCLOSURE in " ".join(text.split())


@pytest.mark.parametrize("via,label", TRANSPORTS)
def test_phone_label(via, label):
    assert phone.link_label(via) == label


@pytest.mark.parametrize("via,label", TRANSPORTS)
def test_review_phone_output_and_page_record(video, binds, monkeypatch, capsys, via, label):
    from yap_studio.review.server import ReviewServer

    monkeypatch.setattr(phone, "tailscale_name", lambda: "synthetic.test" if via == "tailscale"
                        else None)
    monkeypatch.setattr(phone, "tailscale_origin", lambda port: ("https://synthetic.test", None))
    monkeypatch.setattr(phone, "lan_address", lambda: "192.168.1.20")
    records = []
    monkeypatch.setattr(ReviewServer, "serve_forever", lambda self: records.append(self.phone))
    assert cli.main(["review", str(video), "--phone", "--no-open"]) == 0
    assert label in capsys.readouterr().out.splitlines()[0]
    assert records[0]["via"] == via


@pytest.mark.parametrize("via,label", TRANSPORTS)
@pytest.mark.parametrize("detached", [False, True])
def test_home_phone_output(tmp_path, monkeypatch, capsys, via, label, detached):
    from yap_studio.homepage.server import HomeServer

    (tmp_path / "inbox").mkdir()
    monkeypatch.setattr(phone, "tailscale_name", lambda: "synthetic.test" if via == "tailscale"
                        else None)
    monkeypatch.setattr(phone, "tailscale_origin", lambda port: ("https://synthetic.test", None))
    monkeypatch.setattr(phone, "lan_address", lambda: "127.0.0.1")
    monkeypatch.setattr(HomeServer, "serve_forever", lambda self: None)
    if detached:
        home_detach.show({"url": "http://127.0.0.1:12345/", "phone": {
            "via": via, "url": "http://synthetic.test/",
        }}, json_output=False, no_open=True, pid=123)
    else:
        assert cli.main(["home", "--workspace", str(tmp_path), "--phone", "--no-open"]) == 0
    assert label in capsys.readouterr().out


@pytest.mark.parametrize("via,label", TRANSPORTS)
def test_detached_review_phone_output(tmp_path, monkeypatch, capsys, via, label):
    document = {"list": None, "videos": [], "phone": {
        "via": via, "url": "http://synthetic.test/?t=synthetic",
    }}
    monkeypatch.setattr(background, "records", lambda kind: [])
    monkeypatch.setattr(background, "spawn_record", lambda *args, **kwargs: (None, None, None))
    monkeypatch.setattr(background, "ready", lambda *args, **kwargs: {
        "pid": 123, "document": document,
    })
    monkeypatch.setattr(background, "log_lines", lambda log: [])
    args = cli._parser().parse_args(["review", str(tmp_path / "sample.mp4"),
                                    "--phone", "--detach", "--no-open"])
    assert review_detach.start(args) == 0
    assert label in capsys.readouterr().out


@pytest.mark.parametrize("via,label", TRANSPORTS)
def test_detached_style_phone_output(capsys, via, label):
    style_detach.show({"url": "http://synthetic.test/", "phone": {"via": via}},
                      123, json_output=False, no_open=True)
    assert label in capsys.readouterr().out


@pytest.mark.parametrize("via,label", TRANSPORTS)
def test_review_api_carries_phone_transport(removed_server, via, label):
    removed_server.phone = {"via": via, "url": "http://synthetic.test/?t=test-token"}
    code, _, body = _in_process_request(removed_server, "GET", "/api/review")
    assert code == 200
    assert json.loads(body)["phone"] == {
        "via": via, "url": "http://synthetic.test/?t=test-token&version=2",
    }


@pytest.mark.parametrize("via,label", TRANSPORTS)
def test_phone_link_does_not_expose_the_review_list_token(removed_server, via, label):
    server = removed_server
    server.phone = {"via": via, "url": "http://synthetic.test/?t=list-only-token"}
    server.videos = (server.videos[0], replace(server.videos[0], token="second-only-token"))
    code, _, body = _in_process_request(server, "GET", "/v/2/api/review?version=1",
                                       token="second-only-token")
    assert code == 200
    assert json.loads(body)["phone"] == {
        "via": via, "url": "http://synthetic.test/v/2/?t=second-only-token&version=1",
    }
    assert b"list-only-token" not in body


@pytest.mark.parametrize("via,label", TRANSPORTS)
def test_review_list_labels_each_phone_link(removed_server, via, label):
    server = removed_server
    server.phone = {"via": via, "url": "http://synthetic.test/?t=list-only-token"}
    server.list_token = "list-only-token"
    server.videos = (server.videos[0], replace(server.videos[0], token="second-only-token"))
    code, _, body = _in_process_request(server, "GET", "/", token="list-only-token")
    assert code == 200
    assert body.decode().count(label) == 2
    assert b"v/1/?t=test-token" in body
    assert b"v/2/?t=second-only-token" in body
