"""MCP stdio server generated from the operation registry."""

from __future__ import annotations

import inspect
import logging
import sys
import traceback
from collections.abc import Callable, Mapping
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, get_type_hints

from yap_studio.errors import DependencyError, YapError
from yap_studio.registry import REGISTRY, Op

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

_LOGGER = logging.getLogger("yap_studio")
_MCP_INSTALL = "pip install 'yap-studio[mcp]'"


def _missing_mcp() -> DependencyError:
    return DependencyError(
        "MCP support is not installed.",
        error_type="dependency_error",
        code="mcp_not_installed",
        suggested_action=_MCP_INSTALL,
    )


def safe_tool(op: Op) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Wrap an operation so every result is a structured success or error."""

    def call(arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            report = op.call(arguments)
        except YapError as exc:
            return {"success": False, "error": exc.to_dict()}
        except Exception:  # noqa: BLE001 - this is the protocol's final error boundary
            _LOGGER.error("Unexpected error in %s:\n%s", op.name, traceback.format_exc())
            return {
                "success": False,
                "error": {
                    "type": "internal_error",
                    "code": "internal_error",
                    "message": "Unexpected error; see the server log.",
                },
            }
        return {"success": True, **report}

    return call


def _tool_function(op: Op) -> Callable[..., dict[str, Any]]:
    guarded = safe_tool(op)

    def invoke(**arguments: Any) -> dict[str, Any]:
        return guarded(arguments)

    original = inspect.signature(op.fn)
    hints = get_type_hints(op.fn)
    parameters = [
        parameter.replace(annotation=hints[name])
        for name, parameter in original.parameters.items()
        if name not in op.excluded_params
    ]
    invoke.__name__ = op.name
    invoke.__doc__ = f"{op.summary} {op.side_effects}"
    invoke.__signature__ = original.replace(  # type: ignore[attr-defined]
        parameters=parameters,
        return_annotation=dict[str, Any],
    )
    invoke.__annotations__ = {
        **{parameter.name: hints[parameter.name] for parameter in parameters},
        "return": dict[str, Any],
    }
    return invoke


def build_server() -> FastMCP:
    """Build the MCP server from the operation registry."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise _missing_mcp() from exc

    server = FastMCP("yap-studio")
    server._mcp_server.version = version("yap-studio")
    for op in REGISTRY.values():
        server.add_tool(
            _tool_function(op),
            name=op.name,
            description=f"{op.summary} {op.side_effects}",
            structured_output=True,
        )
        server._tool_manager._tools[op.name].parameters = op.input_schema()
    return server


def run_stdio() -> int:
    """Run the MCP server over stdio with application logs confined to stderr."""
    logger = logging.getLogger("yap_studio")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("yap-studio: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    build_server().run("stdio")
    return 0
