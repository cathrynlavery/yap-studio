"""Errors with messages and structured recovery details."""

from __future__ import annotations

import platform
import re
from collections.abc import Sequence


def shorten_text(text: str, limit: int = 240) -> str:
    """Keep both ends of long text with an exact removal count."""
    limit = max(0, limit)
    if len(text) <= limit:
        return text
    removed = len(text) - limit
    while True:
        marker = f" [{removed} chars cut] "
        if len(marker) > limit:
            return text[:limit]
        budget = limit - len(marker)
        actual_removed = len(text) - budget
        if actual_removed == removed:
            start = budget // 3
            end = budget - start
            return text[:start] + marker + (text[-end:] if end else "")
        removed = actual_removed


class YapError(Exception):
    """Carry a message and optional recovery instructions."""

    def __init__(
        self,
        message: str,
        error_type: str = "yap_error",
        code: str = "error",
        *,
        suggested_action: str | None = None,
        docs_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.code = code
        self.suggested_action = suggested_action
        self.docs_url = docs_url

    @property
    def message(self) -> str:
        """Read the message from the current exception arguments."""
        return str(self.args[0]) if self.args else ""

    @message.setter
    def message(self, value: str) -> None:
        self.args = (value,)

    def to_dict(self) -> dict[str, str]:
        """Return current details in the public JSON field order."""
        result = {
            "type": self.error_type,
            "code": self.code,
            "message": shorten_text(self.message, 4000),
        }
        if self.suggested_action is not None:
            result["suggested_action"] = self.suggested_action
        if self.docs_url is not None:
            result["documentation_url"] = self.docs_url
        return result


class DependencyError(YapError):
    """Report a missing or unsupported runtime dependency."""

    @classmethod
    def ffmpeg_too_old(
        cls, name: str, found: str, minimum: str = "6.1"
    ) -> DependencyError:
        """Build an error for a media executable below the supported minimum."""
        commands = [
            ("Darwin", "brew upgrade ffmpeg"),
            (
                "Linux",
                "Ubuntu 24.04 and later ship 6.1; on older releases install a newer static build "
                "and point YAP_STUDIO_FFMPEG / YAP_STUDIO_FFPROBE at it",
            ),
            ("Windows", "winget upgrade ffmpeg"),
        ]
        ordered = sorted(commands, key=lambda item: item[0] != platform.system())
        return cls(
            f"{name} {found} is too old. Yap Studio needs {name} {minimum} or newer.",
            error_type="dependency_error",
            code=f"{name}_too_old",
            suggested_action="; ".join(command for _, command in ordered),
        )

    @classmethod
    def missing_tool(cls, name: str) -> DependencyError:
        """Build an error for an unavailable media executable."""
        commands = [
            ("Darwin", "brew install ffmpeg"),
            ("Linux", "sudo apt-get install ffmpeg"),
            ("Windows", "winget install ffmpeg"),
        ]
        current = platform.system()
        ordered = sorted(commands, key=lambda item: item[0] != current)
        installs = "; ".join(command for _, command in ordered)
        action = (
            f"Install ffmpeg with one of: {installs}; or set "
            "YAP_STUDIO_FFMPEG / YAP_STUDIO_FFPROBE to the executable"
        )
        return cls(
            f"{name} not found.",
            error_type="dependency_error",
            code=f"{name}_not_found",
            suggested_action=action,
        )

    @classmethod
    def missing_engine(
        cls,
        name: str,
        hint: str = "pip install 'yap-studio[asr]'",
    ) -> DependencyError:
        """Build an error for an unavailable speech engine."""
        return cls(
            f"{name} not found.",
            error_type="dependency_error",
            code="speech_engine_not_found",
            suggested_action=f"Install the speech engine, then try again: {hint}.",
        )

    @classmethod
    def missing_node(cls) -> DependencyError:
        """Build an error for unavailable Node tools."""
        return cls(
            "Node not found.",
            error_type="dependency_error",
            code="node_not_found",
            suggested_action="Install Node 22 or newer, then try again.",
        )

    @classmethod
    def missing_renderer(cls) -> DependencyError:
        """Build an error for the unavailable pinned renderer."""
        return cls(
            "Renderer not found.",
            error_type="dependency_error",
            code="renderer_not_found",
            suggested_action="Run yap-studio doctor to install the pinned renderer.",
        )

    @classmethod
    def missing_xcode_tools(cls) -> DependencyError:
        """Build an error for unavailable Xcode command-line tools."""
        return cls(
            "Xcode command-line tools not found.",
            error_type="dependency_error",
            code="xcode_tools_not_found",
            suggested_action="Install the command-line tools with xcode-select --install.",
        )

    @classmethod
    def missing_elevenlabs_cli(cls) -> DependencyError:
        """Build an error for the unavailable ElevenLabs CLI."""
        return cls(
            "ElevenLabs CLI not found.",
            error_type="dependency_error",
            code="elevenlabs_cli_not_found",
            suggested_action=(
                "Install it with: npx -y @mvanhorn/printing-press-library "
                "install elevenlabs --cli-only. Then create your own ElevenLabs API key "
                "with Music, Sound Effects and User read access, and save it: "
                "elevenlabs-pp-cli auth set-token <your-key>"
            ),
        )

    @classmethod
    def missing_pexels_cli(cls) -> DependencyError:
        """Build an error with optional stock setup instructions."""
        from yap_studio.stock.pexels import INSTALL_LINE, KEY_LINE

        return cls("Pexels CLI not found.", error_type="dependency_error",
                   code="pexels_cli_not_found", suggested_action=INSTALL_LINE + "\n" + KEY_LINE)

    @classmethod
    def missing_wavespeed_cli(cls) -> DependencyError:
        """Build an error with optional picture setup instructions."""
        from yap_studio.pictures.wavespeed import INSTALL_LINE, KEY_LINE

        return cls("WaveSpeed CLI not found.", error_type="dependency_error",
                   code="wavespeed_cli_not_found", suggested_action=INSTALL_LINE + "\n" + KEY_LINE)


class RendererBlockedError(YapError):
    """Report a renderer whose browser or worker cannot start inside a sandbox."""

    def __init__(self) -> None:
        super().__init__(
            "The agent sandbox blocked the renderer from starting.",
            error_type="dependency_error",
            code="renderer_blocked",
            suggested_action=(
                "Run the command outside the agent sandbox, for example in a normal terminal."
            ),
        )


class InputError(YapError):
    """Report an input that cannot be read or decoded."""

    def __init__(self, path: str, reason: str | None = None) -> None:
        self.path = str(path)
        self.reason = "no file at this path" if reason is None else reason
        prefix = (
            f"Cannot read input {shorten_text(self.path)}" if self.path else "Cannot read input"
        )
        super().__init__(
            f"{prefix}: {self.reason}",
            error_type="input_error",
            code="invalid_input",
            suggested_action="Check that the path points to an existing audio or video file.",
        )


class ValidationError(YapError, ValueError):
    """Report a parameter and the reason its value is invalid."""

    def __init__(self, parameter: str, detail: str) -> None:
        self.parameter = parameter
        self.detail = detail
        super().__init__(
            f"Bad value for {parameter}: {detail}",
            error_type="validation_error",
            code="invalid_parameter",
            suggested_action=f"Fix {parameter} and retry.",
        )


class EditListError(YapError):
    """Report an invalid edit list."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_edit_list",
        suggested_action: str | None = None,
        path: str | None = None,
    ) -> None:
        self.path = path
        super().__init__(
            message,
            error_type="edit_list_error",
            code=code,
            suggested_action=suggested_action,
        )


_LIBRARY_VERSIONS = re.compile(
    r"lib\w+\s+\d+\.\s*\d+\.\s*\d+\s*/\s*\d+\.\s*\d+\.\s*\d+"
)


def _stderr_tail(stderr: str) -> str:
    lines = []
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith(("ffmpeg version ", "ffprobe version ", "built with ",
                                "configuration:")) or _LIBRARY_VERSIONS.fullmatch(stripped):
            continue
        lines.append(line)
    useful = "\n".join(lines).strip() or stderr.strip()
    return useful[-400:] or "(no error output)"


class ProcessingError(YapError):
    """Report a failed media command while retaining its full output."""

    def __init__(
        self,
        command: str,
        returncode: int,
        stderr: str,
        *,
        code: str | None = None,
        stdout: str | None = None,
    ) -> None:
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.full_stderr = stderr
        super().__init__(
            f"The media tool exited with return code {returncode}. "
            f"Last error output: {_stderr_tail(stderr)}",
            error_type="processing_error",
            code=f"ffmpeg_exit_{returncode}" if code is None else code,
        )


class OutputPathError(YapError):
    """Report an unsafe or invalid output path."""

    def __init__(self, message: str, *, code: str = "invalid_output_path") -> None:
        super().__init__(message, error_type="validation_error", code=code)


def classify_media_failure(stderr: str, command: Sequence[str] | None = None) -> YapError:
    """Classify familiar input failures, otherwise return a processing error."""
    arguments = list(command or [])
    path = ""
    if "-i" in arguments:
        index = arguments.index("-i") + 1
        if index < len(arguments):
            path = arguments[index]
    lowered = stderr.lower()
    if "no such file or directory" in lowered:
        return InputError(path)
    if "invalid data found when processing input" in lowered:
        return InputError(path, "ffmpeg cannot decode it as audio or video")
    if "permission denied" in lowered or "operation not permitted" in lowered:
        return InputError(path, "no permission to read it")
    return ProcessingError(" ".join(arguments), 1, stderr)
