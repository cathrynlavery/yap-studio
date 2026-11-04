"""Voice decisions and recorded audio over HTTP and the in-process dispatcher."""

import copy
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace

import pytest
from test_review_taste import project_server as project_server
from test_review_taste import round_project as round_project
from test_review_taste import socket_access as socket_access
from test_review_taste import taste_client as taste_client

from yap_studio.voice import store
from yap_studio.voice.model import STATEMENT, write
from yap_studio.voice.plan import estimate, list_fixes


@pytest.fixture
def ready_fix(taste_client):
    server, _ = taste_client
    root = server.series.project_dir
    words = server.series.versions[-1].words
    word = words.words[0]
    # The source clock differs from the output clock after a cut.
    shifted = replace(word, start=192.4, end=193.0)
    server.series = replace(server.series, versions=tuple(
        replace(version, words=replace(words, words=(shifted,)))
        for version in server.series.versions
    ))
    files = {}
    for name in ("before", "after", "patch"):
        relative = f"voice/fixes/f1/{name}.wav"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"RIFF" + name.encode() + bytes(range(100)))
        files[name] = {"path": relative, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    fix = {
        "id": "f1", "take": "t1", "words": [word.id], "said": "coal flunge",
        "say": "cold plunge", "why": "Misspoke", "status": "ready",
        "source": {"start_s": 62.4, "end_s": 63.0}, "picture": {"kind": "voiceover"},
        "estimate": {"characters": 11, "credits": 11, "model": "eleven_multilingual_v2",
                     "multiplier": 1, "source": "models"},
        "patch": {**files["patch"], "start_sample": 2995200, "samples": 28800,
                  "channels": 1, "fade_in_samples": 720, "fade_out_samples": 720,
                  "stretch": 1.04, "gain_db": 0},
        "clips": {name: files[name] for name in ("before", "after")},
        "generation": {"request_sha256": "a" * 64, "output_sha256": "b" * 64,
                       "seed": 1, "model": "eleven_multilingual_v2", "at": "now",
                       "cost": {"credits": 9, "source": "history"}},
        "checks": {"status": "pass", "items": [
            {"id": "splice.click", "status": "pass"},
            {"id": "splice.level", "status": "pass", "difference_lu": 0.6,
             "reference_lufs": -20, "reference_rms_db": -20},
            {"id": "splice.room_tone", "status": "pass", "source_floor_db": -50},
            {"id": "splice.pace", "status": "pass", "stretch": 1.04, "gap_s": 0},
            {"id": "splice.heard", "status": "pass", "heard": "cold plunge"},
        ]},
    }
    write(root, "fixes", {"schema": "yap-studio/voice-fixes@1", "fixes": [fix]})
    samples = [{"take": "t1", "path": f"voice/samples/s{i:02}.wav", "sha256": "c" * 64,
                "start_s": (i - 1) * 15, "end_s": i * 15} for i in range(1, 5)]
    write(root, "consent", {"schema": "yap-studio/voice-consent@1", "status": "given",
          "statement": STATEMENT, "given_by": "Creator", "given_at": "now", "samples": samples,
          "voice": {"name": "My voice", "id_sha256": "d" * 64, "made": "linked", "at": "now"}})
    return fix


def save_fixes(server, *fixes):
    write(server.series.project_dir, "fixes",
          {"schema": "yap-studio/voice-fixes@1", "fixes": list(fixes)})


def test_review_own_source_and_generation_clips(taste_client, ready_fix):
    from yap_studio import receipt

    server, send = taste_client
    root = server.series.project_dir
    ready_fix["from"] = {"kind": "own", "take": "t1", "words": ["t1:9", "t1:10"],
                         "start_s": 208, "end_s": 209, "text": "cold plunge"}
    ready_fix["estimate"]["credits"] = ready_fix["generation"]["cost"]["credits"] = 0
    for row in (ready_fix["patch"], *ready_fix["clips"].values()):
        old = root / row["path"]
        new = old.parent / "g0123456789abcdef" / old.name
        new.parent.mkdir(exist_ok=True)
        old.rename(new)
        row.update(path=str(new.relative_to(root)), sha256=receipt.sha256_file(new))
    save_fixes(server, ready_fix)
    result = json.loads(send("GET", "/api/review")[2])["series"]["voice_fixes"][0]
    assert result["kind"] == "own" and result["from_time"] == "3:28"
    assert result["say"] == "cold plunge" and result["credits"] == 0
    for name in ("before", "after"):
        assert send("GET", f"/voice-fix-audio?fix=f1&clip={name}")[0] == 200


def fingerprint(fix):
    return hashlib.sha256(
        f"{fix['generation']['output_sha256']}:{fix['patch']['sha256']}".encode()
    ).hexdigest()


def test_voice_json_shape_and_output_time(taste_client, ready_fix):
    _, send = taste_client
    status, _, body = send("GET", "/api/review")
    assert status == 200
    fix = json.loads(body)["series"]["voice_fixes"][0]
    assert fix == {
        "kind": "clone", "from_time": None,
        "id": "f1", "said": "coal flunge", "say": "cold plunge", "time_s": 192.4,
        "time": "3:12", "generation": fingerprint(ready_fix),
        "before_url": f"voice-fix-audio?fix=f1&clip=before&generation={fingerprint(ready_fix)}",
        "after_url": f"voice-fix-audio?fix=f1&clip=after&generation={fingerprint(ready_fix)}",
        "checks": ["Join: clean.", "Loudness: within 0.6 LU of your voice.",
                   "Room tone: matched.", "Pace: 4% slower to fit.", "Heard as: cold plunge."],
        "credits": 9, "credits_source": "charged", "cost_note": "",
        "status": "ready", "decision": None,
    }


@pytest.mark.parametrize("changed", ["output", "patch"])
@pytest.mark.parametrize("action", ["approve", "reject"])
def test_decision_refuses_a_generation_the_creator_did_not_hear(
    taste_client, ready_fix, changed, action,
):
    server, send = taste_client
    generation = fingerprint(ready_fix)
    if changed == "output":
        ready_fix["generation"]["output_sha256"] = "e" * 64
    else:
        ready_fix["patch"]["sha256"] = "e" * 64
    save_fixes(server, ready_fix)
    status, _, body = send("POST", "/api/voice-fixes", {
        "fix": "f1", "action": action, "generation": generation,
    })
    assert status == 409
    error = json.loads(body)["error"]
    assert error["code"] == "voice_fix_changed"
    assert error["message"] == "This fix was made again. Listen to the new version first."
    assert list_fixes(server.series.project_dir)["fixes"][0].get("decision") is None
    fresh = json.loads(send("GET", "/api/review")[2])["series"]["voice_fixes"][0]
    assert fresh["generation"] != generation
    for clip in ("before", "after"):
        assert f"generation={fresh['generation']}" in fresh[f"{clip}_url"]
        assert send("GET", f"/voice-fix-audio?fix=f1&clip={clip}&generation={generation}")[0] == 409


def test_generation_is_checked_inside_the_decision_lock(taste_client, ready_fix, monkeypatch):
    server, send = taste_client
    generation = fingerprint(ready_fix)
    project_lock = store.project_lock

    @contextmanager
    def remake_before_lock(root):
        with project_lock(root):
            ready_fix["generation"]["output_sha256"] = "e" * 64
            save_fixes(server, ready_fix)
            yield

    monkeypatch.setattr(store, "project_lock", remake_before_lock)
    status, _, body = send("POST", "/api/voice-fixes", {
        "fix": "f1", "action": "approve", "generation": generation,
    })
    assert status == 409 and json.loads(body)["error"]["code"] == "voice_fix_changed"
    assert list_fixes(server.series.project_dir)["fixes"][0].get("decision") is None


def test_empty_and_listenable_states(taste_client, ready_fix):
    server, send = taste_client
    root = server.series.project_dir
    (root / "voice/fixes.json").unlink()
    assert json.loads(send("GET", "/api/review")[2])["series"]["voice_fixes"] == []
    fixes = []
    for i, state in enumerate(("ready", "approved", "rejected", "proposed", "failed"), 1):
        fix = copy.deepcopy(ready_fix)
        fix.update(id=f"f{i}", status=state)
        for row in (fix["patch"], *fix["clips"].values()):
            row["path"] = row["path"].replace("/f1/", f"/f{i}/")
        fixes.append(fix)
    save_fixes(server, *fixes)
    assert [fix["status"] for fix in json.loads(send("GET", "/api/review")[2])[
        "series"]["voice_fixes"]] == ["ready", "approved", "rejected"]


def test_estimated_cost_and_short_note(taste_client, ready_fix):
    server, send = taste_client
    ready_fix["generation"]["cost"] = {"credits": 11, "source": "estimate"}
    ready_fix["estimate"]["unavailable"] = {"models": "models_read", "plan": "user_read"}
    save_fixes(server, ready_fix)
    fix = json.loads(send("GET", "/api/review")[2])["series"]["voice_fixes"][0]
    assert (fix["credits"], fix["credits_source"]) == (11, "estimated")
    assert fix["cost_note"] == "Credits left could not be read"
    assert "models_read" not in json.dumps(fix) and "user_read" not in json.dumps(fix)


def test_decisions_can_be_changed_and_use_review(taste_client, ready_fix):
    server, send = taste_client
    for action, state in (("approve", "approved"), ("reject", "rejected"), ("approve", "approved")):
        status, _, body = send("POST", "/api/voice-fixes", {
            "fix": "f1", "action": action, "generation": fingerprint(ready_fix),
        })
        assert status == 200, body
        saved = list_fixes(server.series.project_dir)["fixes"][0]
        assert saved["status"] == state
        assert saved["decision"]["action"] == action and saved["decision"]["via"] == "review"
        assert json.loads(body)["voice_fixes"][0]["decision"] == saved["decision"]


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_voice_decision_keeps_selected_version_clock(taste_client, ready_fix, action):
    server, send = taste_client
    server.series = replace(server.series, versions=tuple(
        replace(version, words=replace(version.words, words=tuple(
            replace(word, start=12.4, end=13.0) for word in version.words.words
        ))) if version.number == 1 else version
        for version in server.series.versions
    ))
    before = json.loads(send("GET", "/api/review?version=1")[2])["series"]["voice_fixes"][0]
    assert (before["time_s"], before["time"]) == (12.4, "0:12")
    status, _, body = send("POST", "/api/voice-fixes?version=1", {
        "fix": "f1", "action": action, "generation": fingerprint(ready_fix),
    })
    assert status == 200, body
    saved = json.loads(body)["voice_fixes"][0]
    assert (saved["time_s"], saved["time"]) == (before["time_s"], before["time"])
    latest = json.loads(send("GET", "/api/review?version=2")[2])["series"]["voice_fixes"][0]
    assert (latest["time_s"], latest["time"]) == (192.4, "3:12")


@pytest.mark.parametrize("changed_version", [1, 2])
def test_voice_routes_guard_only_changed_version(taste_client, ready_fix, changed_version):
    server, send = taste_client
    server.series.versions[changed_version - 1].media.path.write_bytes(b"changed render")
    body = {"fix": "f1", "action": "approve", "generation": fingerprint(ready_fix)}
    for method, route, payload in (
        ("POST", "/api/voice-fixes", body),
        ("GET", "/voice-fix-audio?fix=f1&clip=before", None),
    ):
        separator = "&" if "?" in route else "?"
        status, _, response = send(method, f"{route}{separator}version={changed_version}", payload)
        assert status == 409, response
        assert json.loads(response)["error"]["code"] == "media_changed"
    assert list_fixes(server.series.project_dir)["fixes"][0].get("decision") is None
    assert send("POST", f"/api/voice-fixes?version={3 - changed_version}", body)[0] == 200


@pytest.mark.parametrize("refusal", ["consent", "patch", "not_ready", "checks"])
def test_refused_approval_is_409(taste_client, ready_fix, refusal):
    server, send = taste_client
    root = server.series.project_dir
    if refusal == "consent":
        record = json.loads((root / "voice/consent.json").read_text())
        record.update(status="withdrawn", withdrawn_at="now")
        write(root, "consent", record)
    elif refusal == "patch":
        (root / ready_fix["patch"]["path"]).write_bytes(b"changed")
    else:
        if refusal == "not_ready":
            ready_fix["status"] = "proposed"
        else:
            ready_fix["checks"]["status"] = "fail"
        save_fixes(server, ready_fix)
    status, _, body = send("POST", "/api/voice-fixes", {
        "fix": "f1", "action": "approve", "generation": fingerprint(ready_fix),
    })
    assert status == 409
    assert json.loads(body)["error"]["message"]
    assert list_fixes(root)["fixes"][0].get("decision") is None


def test_unknown_fix_and_invalid_decisions(taste_client, ready_fix):
    _, send = taste_client
    generation = fingerprint(ready_fix)
    assert send("POST", "/api/voice-fixes", {
        "fix": "f9", "action": "approve", "generation": generation,
    })[0] == 404
    for body in ({}, {"fix": "f1", "action": "approve"},
                 {"fix": "f1", "action": "bad", "generation": generation},
                 {"fix": [], "action": "approve", "generation": generation},
                 {"fix": "f1", "action": "reject", "via": "chat", "generation": generation},
                 {"fix": "f1", "action": "approve", "generation": []},
                 {"fix": "f1", "action": "approve", "generation": ""}):
        assert send("POST", "/api/voice-fixes", body)[0] == 400


def test_voice_route_auth_and_origin(taste_client, ready_fix):
    _, send = taste_client
    body = {"fix": "f1", "action": "approve", "generation": fingerprint(ready_fix)}
    assert send("POST", "/api/voice-fixes", body, token=None)[0] == 403
    assert send("POST", "/api/voice-fixes?t=test-token", body, token=None)[0] == 403
    assert send("POST", "/api/voice-fixes", body, host="evil.test")[0] == 403
    assert send("GET", "/api/voice-fixes")[0] == 405
    route = "/voice-fix-audio?fix=f1&clip=before"
    assert send("GET", route, token=None)[0] == 403
    assert send("GET", route + "&t=test-token", token=None)[0] == 200


def test_audio_recorded_files_hash_and_range(taste_client, ready_fix):
    server, send = taste_client
    root = server.series.project_dir
    for clip in ("before", "after"):
        route = f"/voice-fix-audio?fix=f1&clip={clip}"
        expected = (root / ready_fix["clips"][clip]["path"]).read_bytes()
        status, headers, body = send("GET", route)
        assert status == 200 and body == expected and headers["Content-Type"] == "audio/wav"
        for span, selected in (("bytes=2-9", expected[2:10]), ("bytes=-4", expected[-4:]),
                               ("bytes=4-", expected[4:])):
            status, headers, body = send("GET", route, headers={"Range": span})
            assert status == 206 and body == selected and headers["Accept-Ranges"] == "bytes"
            assert headers["Content-Range"].endswith(f"/{len(expected)}")
        assert send("GET", route, headers={"Range": "bytes=999-"})[0] == 416
        assert send("HEAD", route)[2] == b""
        (root / ready_fix["clips"][clip]["path"]).write_bytes(b"changed")
        assert send("GET", route)[0] == 409
    assert send("GET", "/voice-fix-audio?fix=f9&clip=before")[0] == 404
    assert send("GET", "/voice-fix-audio?fix=f1&clip=patch")[0] == 400
    assert send("GET", "/voice-fix-audio?fix=../../secret&clip=before")[0] == 404


@pytest.mark.parametrize("escape", ["record", "symlink"])
def test_audio_cannot_escape_fix_directory(taste_client, ready_fix, escape):
    server, send = taste_client
    root = server.series.project_dir
    target = root / ready_fix["clips"]["before"]["path"]
    outside = root / "private.wav"
    outside.write_bytes(target.read_bytes())
    if escape == "record":
        ready_fix["clips"]["before"]["path"] = "private.wav"
        data = {"schema": "yap-studio/voice-fixes@1", "fixes": [ready_fix]}
        (root / "voice/fixes.json").write_text(json.dumps(data))
    else:
        target.unlink()
        target.symlink_to(outside)
    status, _, body = send("GET", "/voice-fix-audio?fix=f1&clip=before")
    assert status == 409 and body != outside.read_bytes()


def test_estimate_keeps_only_short_missing_reasons():
    budget = {"multiplier": 1, "source": "estimate", "unavailable": {
        "models": {"code": "elevenlabs_permission_missing", "permission": "models_read",
                   "message": "Long message", "suggested_action": "Long advice"},
        "plan": {"code": "elevenlabs_read_failed", "message": "Long message"},
    }}
    assert estimate("cold plunge", budget)["unavailable"] == {
        "models": "models_read", "plan": "elevenlabs_read_failed",
    }
