"""Operation registry and derived JSON parameter schemas."""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from yap_studio.errors import ValidationError
from yap_studio.verify import verify

_EMPTY = inspect.Parameter.empty


def _json_scalar_type(annotation: Any) -> str:
    mapping = {str: "string", float: "number", int: "integer", bool: "boolean"}
    try:
        return mapping[annotation]
    except (KeyError, TypeError) as exc:
        raise TypeError(f"unsupported operation parameter type: {annotation!r}") from exc


def _type_details(annotation: Any) -> tuple[str, bool, tuple[Any, ...] | None, str | None]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    nullable = False
    if origin in (Union, types.UnionType):
        non_null = tuple(item for item in args if item is not type(None))
        if len(non_null) != 1 or len(non_null) == len(args):
            raise TypeError(f"unsupported operation parameter type: {annotation!r}")
        annotation = non_null[0]
        origin = get_origin(annotation)
        args = get_args(annotation)
        nullable = True

    if origin is Literal:
        if not args:
            raise TypeError("operation Literal parameter must contain at least one value")
        json_type = _json_scalar_type(type(args[0]))
        if any(_json_scalar_type(type(value)) != json_type for value in args[1:]):
            raise TypeError(f"mixed JSON types in operation Literal: {annotation!r}")
        return json_type, nullable, tuple(args), None

    if origin is list:
        if args != (str,):
            raise TypeError(f"unsupported operation parameter type: {annotation!r}")
        return "array", nullable, None, "string"

    return _json_scalar_type(annotation), nullable, None, None


@dataclass(frozen=True)
class Param:
    """One operation parameter derived from a callable signature."""

    name: str
    json_type: str
    required: bool
    default: Any
    nullable: bool
    enum: tuple[Any, ...] | None
    items: str | None
    description: str

    def json_schema(self) -> dict:
        """Return this parameter as JSON Schema."""
        value_schema: dict[str, Any] = {"type": self.json_type}
        if self.enum is not None:
            value_schema["enum"] = list(self.enum)
        if self.items is not None:
            value_schema["items"] = {"type": self.items}

        if self.nullable:
            schema: dict[str, Any] = {
                "anyOf": [value_schema, {"type": "null"}],
                "description": self.description,
            }
        else:
            schema = {**value_schema, "description": self.description}
        if not self.required:
            schema["default"] = self.default
        return schema


@dataclass(frozen=True)
class Op:
    """A callable operation and all of its public metadata."""

    name: str
    fn: Callable[..., dict]
    summary: str
    side_effects: str
    excluded_params: frozenset[str] = frozenset()

    def params(self) -> list[Param]:
        """Derive operation parameters from the function signature and docstring."""
        signature = inspect.signature(self.fn)
        hints = get_type_hints(self.fn)
        descriptions = parse_args_block(inspect.getdoc(self.fn))
        result = []
        for name, parameter in signature.parameters.items():
            if name in self.excluded_params:
                continue
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise TypeError(f"unsupported parameter kind for {name}: {parameter.kind}")
            if name not in hints:
                raise TypeError(f"operation parameter {name} has no type annotation")
            if name not in descriptions:
                raise ValueError(f"operation parameter {name} has no Args description")
            json_type, nullable, enum, items = _type_details(hints[name])
            result.append(
                Param(
                    name=name,
                    json_type=json_type,
                    required=parameter.default is _EMPTY,
                    default=parameter.default,
                    nullable=nullable,
                    enum=enum,
                    items=items,
                    description=descriptions[name],
                )
            )
        return result

    def input_schema(self) -> dict:
        """Return the complete JSON input schema for this operation."""
        params = self.params()
        return {
            "type": "object",
            "properties": {param.name: param.json_schema() for param in params},
            "required": [param.name for param in params if param.required],
            "additionalProperties": False,
        }

    def call(self, arguments: Mapping[str, Any]) -> dict:
        """Validate argument names, fill defaults, and call the operation."""
        params = self.params()
        allowed = {param.name for param in params}
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValidationError("arguments", f"unknown parameter(s): {', '.join(unknown)}")

        values = {}
        for param in params:
            if param.name in arguments:
                values[param.name] = arguments[param.name]
            elif param.required:
                raise ValidationError(param.name, "is required")
            else:
                values[param.name] = param.default
        return self.fn(**values)


def parse_args_block(docstring: str | None) -> dict[str, str]:
    """Parse parameter descriptions from a Google-style ``Args:`` section."""
    if not docstring:
        return {}
    lines = inspect.cleandoc(docstring).splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == "Args:")
    except StopIteration:
        return {}

    descriptions: dict[str, str] = {}
    current: str | None = None
    entry_indent = 0
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            break
        stripped = line.strip()
        name, separator, description = stripped.partition(":")
        if separator and name.isidentifier():
            current = name
            entry_indent = indent
            descriptions[current] = description.strip()
        elif current is not None and indent > entry_indent:
            descriptions[current] = f"{descriptions[current]} {stripped}".strip()
    return descriptions


_VERIFY_DOC = inspect.getdoc(verify) or ""
REGISTRY: dict[str, Op] = {
    "verify": Op(
        "verify",
        verify,
        summary=_VERIFY_DOC.splitlines()[0],
        side_effects=(
            "Reads the video, edit list and sources. Writes only the report and artifacts paths "
            "when given."
        ),
    )
}

# Validate registered signatures at import time so unsupported future parameters fail loudly.
for _registered_op in REGISTRY.values():
    _registered_op.params()
del _registered_op
