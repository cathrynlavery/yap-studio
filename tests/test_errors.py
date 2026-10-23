"""Structured error tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from yap_studio.errors import (
    DependencyError,
    EditListError,
    InputError,
    ProcessingError,
    RendererBlockedError,
    ValidationError,
    YapError,
)


@pytest.mark.parametrize("name", ["ffmpeg", "ffprobe"])
def test_ffmpeg_too_old_has_stable_details(name: str) -> None:
    error = DependencyError.ffmpeg_too_old(name, "4.4.2")
    assert error.message == f"{name} 4.4.2 is too old. Yap Studio needs {name} 6.1 or newer."
    assert error.code == f"{name}_too_old"
    assert error.error_type == "dependency_error"
    assert error.to_dict()["code"] == error.code
    assert "7.0 or newer" in DependencyError.ffmpeg_too_old(name, "6.1", "7.0").message


@pytest.mark.parametrize(
    ("system", "first"),
    [
        ("Darwin", "brew upgrade ffmpeg"),
        ("Linux", "Ubuntu 24.04 and later ship 6.1"),
        ("Windows", "winget upgrade ffmpeg"),
    ],
)
def test_ffmpeg_too_old_orders_upgrade_hints(
    monkeypatch: pytest.MonkeyPatch, system: str, first: str
) -> None:
    monkeypatch.setattr("yap_studio.errors.platform.system", lambda: system)
    action = DependencyError.ffmpeg_too_old("ffmpeg", "5.1.6").suggested_action or ""
    assert action.startswith(first)
    assert "brew upgrade ffmpeg" in action
    assert "winget upgrade ffmpeg" in action
    assert "YAP_STUDIO_FFMPEG / YAP_STUDIO_FFPROBE" in action


@pytest.mark.parametrize("name", ["ffmpeg", "ffprobe"])
@pytest.mark.parametrize(
    ("target", "args"),
    [
        ("yap_studio.cli.verify", ["verify", "input.mp4", "--json"]),
        ("yap_studio.stages.cut.cut", ["cut", "--json"]),
        ("yap_studio.stages.captions.caption_clip",
         ["captions", "input.mp4", "--out", "output.mp4", "--json"]),
    ],
)
def test_ffmpeg_too_old_maps_to_exit_code_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    name: str, target: str, args: list[str],
) -> None:
    import json

    from yap_studio.cli import main

    error = DependencyError.ffmpeg_too_old(name, "5.1.6")

    def refuse(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(target, refuse)
    assert main(args) == 2
    structured = json.loads(capsys.readouterr().err)["error"]
    assert structured["type"] == "dependency_error"
    assert structured == error.to_dict()


@pytest.mark.parametrize("keywords", [False, True])
def test_base_error_construction_and_live_attributes(keywords: bool) -> None:
    if keywords:
        error = YapError(message="initial", error_type="kind", code="code",
                         suggested_action="repair", docs_url="https://example.invalid/help")
    else:
        error = YapError("initial", "kind", "code", suggested_action="repair",
                         docs_url="https://example.invalid/help")
    assert isinstance(error, Exception)
    assert error.message == str(error) == "initial"
    assert error.args == ("initial",)
    assert error.to_dict() == {
        "type": "kind", "code": "code", "message": "initial",
        "suggested_action": "repair", "documentation_url": "https://example.invalid/help",
    }
    assert list(error.to_dict()) == [
        "type", "code", "message", "suggested_action", "documentation_url",
    ]
    error.message = "changed"
    error.error_type = "new_kind"
    error.code = "new_code"
    error.suggested_action = "new repair"
    error.docs_url = ""
    assert str(error) == "changed"
    assert error.args == ("changed",)
    assert error.to_dict() == {
        "type": "new_kind", "code": "new_code", "message": "changed",
        "suggested_action": "new repair", "documentation_url": "",
    }
    error.args = ("shown",)
    assert str(error) == "shown"
    assert error.message == error.to_dict()["message"] == "shown"
    error.suggested_action = None
    error.docs_url = None
    assert list(error.to_dict()) == ["type", "code", "message"]
    assert YapError("plain").to_dict() == {
        "type": "yap_error", "code": "error", "message": "plain",
    }


def test_error_message_handles_empty_and_nontext_args() -> None:
    error = YapError("initial")
    error.args = ()
    assert error.message == str(error) == error.to_dict()["message"] == ""
    error.args = (17,)
    assert error.message == str(error) == error.to_dict()["message"] == "17"


def test_processing_subclass_replaces_args_for_message_and_json() -> None:
    class ContextualProcessingError(ProcessingError):
        def __init__(self) -> None:
            ProcessingError.__init__(self, "ffmpeg -i take.mov", 7, "synthetic stderr")
            self.args = ("new context from subclass",)

    error = ContextualProcessingError()
    assert error.message == str(error) == error.to_dict()["message"] == "new context from subclass"
    assert error.command == "ffmpeg -i take.mov" and error.returncode == 7
    assert error.full_stderr == "synthetic stderr" and error.code == "ffmpeg_exit_7"


def test_error_dict_caps_message_but_keeps_original() -> None:
    import re

    message = "A" * 10000
    error = YapError(message)
    shortened = error.to_dict()["message"]
    assert len(shortened) <= 4000
    marker = re.search(r" \[(\d+) chars cut\] ", shortened)
    assert marker is not None
    kept = len(shortened) - len(marker[0])
    assert int(marker[1]) == len(message) - kept
    assert error.message == str(error) == message


def test_validation_error_contract() -> None:
    error = ValidationError("start", "must be positive")
    assert isinstance(error, YapError) and isinstance(error, ValueError)
    assert (error.parameter, error.detail) == ("start", "must be positive")
    assert error.error_type == "validation_error" and error.code == "invalid_parameter"
    assert str(error) == "Bad value for start: must be positive"
    assert error.suggested_action == "Fix start and retry."


def test_edit_list_error_keeps_details() -> None:
    error = EditListError("bad edit", path="edit.json", suggested_action="Check edit.")
    assert isinstance(error, YapError)
    assert error.path == "edit.json" and error.code == "invalid_edit_list"
    assert error.message == "bad edit" and error.suggested_action == "Check edit."


def test_input_error_contract() -> None:
    from yap_studio.errors import shorten_text

    path = "/media/" + "folder/" * 100 + "take.mov"
    error = InputError(path)
    assert isinstance(error, YapError)
    assert error.path == path and error.reason == "no file at this path"
    assert error.error_type == "input_error" and error.code == "invalid_input"
    assert str(error) == f"Cannot read input {shorten_text(path)}: no file at this path"
    assert "chars cut" in error.message and error.message.endswith("take.mov: no file at this path")
    assert error.suggested_action == (
        "Check that the path points to an existing audio or video file."
    )
    assert str(InputError("", "reason")) == "Cannot read input: reason"
    assert InputError("take.mov", None).reason == "no file at this path"


@pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
def test_processing_error_strips_realistic_preamble(tool: str) -> None:
    preamble = (
        f"  {tool} version 6.1 Copyright (c) the FFmpeg developers\n"
        "  built with Apple clang version 15.0.0\n"
        "  configuration: --enable-shared --enable-gpl\n"
        "  libavutil      58. 29.100 / 58. 29.100\n"
        "  libavcodec     60. 31.102 / 60. 31.102\n"
    )
    stderr = preamble + "\n  Cannot decode synthetic input  \n"
    error = ProcessingError("ffmpeg -i take.mov", 7, stderr, stdout="partial")
    assert isinstance(error, YapError)
    assert (error.command, error.returncode, error.stdout) == (
        "ffmpeg -i take.mov", 7, "partial",
    )
    assert error.full_stderr == stderr
    assert error.error_type == "processing_error" and error.code == "ffmpeg_exit_7"
    assert error.message == (
        "The media tool exited with return code 7. Last error output: Cannot decode synthetic input"
    )
    assert ProcessingError("tool", 2, preamble).message.endswith(preamble.strip()[-400:])


def test_processing_error_tail_empty_output_and_explicit_code() -> None:
    error = ProcessingError("tool", -9, "x" * 600 + "end", code="stopped")
    assert error.code == "stopped" and error.stdout is None
    assert error.message == (
        "The media tool exited with return code -9. Last error output: " + "x" * 397 + "end"
    )
    assert ProcessingError("tool", 1, " \n").message.endswith("(no error output)")
    assert ProcessingError("tool", 1, "", code="").code == ""
    # A library name in actual error text is not a version preamble.
    assert ProcessingError("tool", 1, "libavcodec failed").message.endswith("libavcodec failed")


@pytest.mark.parametrize("limit", [20, 21, 40, 240, 4000])
def test_shorten_text_bound_exact_count_and_balance(limit: int) -> None:
    import re

    from yap_studio.errors import shorten_text

    text = "/media/" + "nested/" * 1000 + "recognisable-take.mov"
    result = shorten_text(text, limit)
    assert len(result) <= limit
    marker = re.search(r" \[(\d+) chars cut\] ", result)
    assert marker is not None
    start, end = result.split(marker[0])
    assert int(marker[1]) == len(text) - len(start) - len(end)
    assert text.startswith(start) and text.endswith(end)
    assert abs(len(end) - 2 * len(start)) <= 2
    if limit >= 240:
        assert result.endswith("recognisable-take.mov")


@pytest.mark.parametrize("text,limit", [("abc", 3), ("abc", 4), ("abcdef", 2), ("abc", 0)])
def test_shorten_text_unchanged_and_tiny_limits(text: str, limit: int) -> None:
    from yap_studio.errors import shorten_text

    assert shorten_text(text, limit) == text[:limit]
    assert shorten_text("short") == "short"


@pytest.mark.parametrize("stderr,reason", [
    ("nO SuCh FiLe Or DiReCtOrY", "no file at this path"),
    ("iNvAlId DaTa FoUnD wHeN pRoCeSsInG iNpUt", "ffmpeg cannot decode it as audio or video"),
    ("pErMiSsIoN dEnIeD", "no permission to read it"),
    ("oPeRaTiOn NoT pErMiTtEd", "no permission to read it"),
])
@pytest.mark.parametrize("command,path", [
    (["ffmpeg", "-i", "first.mov", "-i", "second.mov"], "first.mov"),
    (["ffprobe", "take.mov"], ""),
    (["ffmpeg", "-i"], ""),
    (None, ""),
])
def test_classifier_input_errors(stderr: str, reason: str,
                                 command: list[str] | None, path: str) -> None:
    from yap_studio.errors import classify_media_failure

    error = classify_media_failure(stderr, command)
    assert isinstance(error, InputError)
    assert error.path == path and error.reason == reason


def test_classifier_order_and_fallback() -> None:
    from yap_studio.errors import classify_media_failure

    error = classify_media_failure(
        "Permission denied; Invalid data found when processing input; No such file or directory"
    )
    assert isinstance(error, InputError) and error.reason == "no file at this path"
    error = classify_media_failure("Permission denied; Invalid data found when processing input")
    assert isinstance(error, InputError)
    assert error.reason == "ffmpeg cannot decode it as audio or video"
    for command in (None, ["ffmpeg", "-i", "take.mov"]):
        error = classify_media_failure("other failure\n", command)
        assert isinstance(error, ProcessingError)
        assert error.command == " ".join(command or [])
        assert error.returncode == 1 and error.full_stderr == "other failure\n"


def test_renderer_blocked_error_dict_has_code_and_fix() -> None:
    assert RendererBlockedError().to_dict() == {
        "type": "dependency_error",
        "code": "renderer_blocked",
        "message": "The agent sandbox blocked the renderer from starting.",
        "suggested_action": (
            "Run the command outside the agent sandbox, for example in a normal terminal."
        ),
    }


@pytest.mark.parametrize(
    ("constructor", "code", "hint"),
    [
        (DependencyError.missing_node, "node_not_found", "Node 22 or newer"),
        (DependencyError.missing_renderer, "renderer_not_found", "yap-studio doctor"),
        (DependencyError.missing_xcode_tools, "xcode_tools_not_found", "xcode-select --install"),
        (DependencyError.missing_elevenlabs_cli, "elevenlabs_cli_not_found", "auth set-token"),
    ],
)
def test_new_dependency_errors_have_stable_codes_and_install_hints(
    constructor: Callable[[], DependencyError], code: str, hint: str
) -> None:
    error = constructor()
    assert error.error_type == "dependency_error"
    assert error.code == code
    assert hint in (error.suggested_action or "")
    assert error.to_dict()["suggested_action"] == error.suggested_action
    if code == "elevenlabs_cli_not_found":
        assert "install elevenlabs --cli-only" in error.suggested_action
