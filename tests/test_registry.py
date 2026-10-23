"""Operation registry schema and drift tests."""

from __future__ import annotations

import argparse
import inspect

import pytest

from yap_studio.cli import _parser
from yap_studio.errors import ValidationError
from yap_studio.registry import REGISTRY, Op
from yap_studio.verify import verify


def test_verify_params_match_signature() -> None:
    op = REGISTRY["verify"]
    expected = [
        parameter
        for name, parameter in inspect.signature(verify).parameters.items()
        if name not in op.excluded_params
    ]
    actual = op.params()
    assert [param.name for param in actual] == [parameter.name for parameter in expected]
    assert [param.required for param in actual] == [
        parameter.default is inspect.Parameter.empty for parameter in expected
    ]
    assert [param.default for param in actual] == [parameter.default for parameter in expected]


def test_verify_registry_keeps_public_parameters_without_transitions() -> None:
    """Internal transition support must not change the registered operation schema."""
    op = REGISTRY["verify"]
    expected = {
        "media": {"type": "string"},
        "edl": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "source": {"anyOf": [{"type": "array", "items": {"type": "string"}},
                             {"type": "null"}], "default": None},
        "asr": {"type": "string",
                "enum": ["auto", "parakeet-mlx", "parakeet-onnx", "faster-whisper", "none"],
                "default": "auto"},
        "asr_model": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "loudness_target": {"type": "number", "default": -14.0},
        "strict": {"type": "boolean", "default": False},
        "report": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "artifacts": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "detail": {"type": "string", "enum": ["summary", "full"], "default": "summary"},
        "platform": {"type": "string", "enum": ["none", "youtube", "shorts", "tiktok", "reels"],
                     "default": "none"},
        "loudness": {"type": "string", "enum": ["check", "note"], "default": "check"},
    }
    assert op.fn is verify
    assert list(inspect.signature(verify).parameters) == list(expected)
    schema = op.input_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["media"]
    assert schema["additionalProperties"] is False
    assert list(schema["properties"]) == list(expected)
    assert {
        name: {key: value for key, value in property_schema.items() if key != "description"}
        for name, property_schema in schema["properties"].items()
    } == expected


def test_verify_params_derive_literals_arrays_and_descriptions() -> None:
    params = {param.name: param for param in REGISTRY["verify"].params()}
    assert params["asr"].enum == (
        "auto",
        "parakeet-mlx",
        "parakeet-onnx",
        "faster-whisper",
        "none",
    )
    assert params["source"].json_type == "array"
    assert params["source"].items == "string"
    assert params["source"].nullable is True
    assert params["loudness"].enum == ("check", "note")
    assert params["loudness"].default == "check"
    assert "true peak" in params["loudness"].description
    assert "intermediate render" in params["loudness"].description
    assert all(param.description for param in params.values())


def test_unsupported_parameter_type_fails_loudly() -> None:
    def unsupported(value: dict[str, str]) -> dict:
        """Use an unsupported value.

        Args:
            value: Unsupported mapping.
        """
        return value

    with pytest.raises(TypeError, match="unsupported operation parameter type"):
        Op("unsupported", unsupported, "Unsupported.", "None.").params()


def test_call_rejects_unknown_argument() -> None:
    with pytest.raises(ValidationError, match="unknown parameter"):
        REGISTRY["verify"].call({"media": "video.mp4", "surprise": True})


def test_cli_covers_every_registry_parameter() -> None:
    parser = _parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    verify_parser = subparsers.choices["verify"]
    actions = {action.dest: action for action in verify_parser._actions}
    assert actions["loudness"].choices == ("check", "note")
    assert actions["loudness"].default == "check"
    for param in REGISTRY["verify"].params():
        assert param.name in actions
        if param.name == "media":
            assert actions[param.name].option_strings == []
        else:
            assert f"--{param.name.replace('_', '-')}" in actions[param.name].option_strings
