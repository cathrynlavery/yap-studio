"""WaveSpeed CLI adapter, with credentials left in the creator's CLI."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

from yap_studio import tools
from yap_studio.affiliate import WAVESPEED_AFFILIATE_URL
from yap_studio.errors import DependencyError, ProcessingError, YapError

PROVIDER = "wavespeed"
LABEL = "WaveSpeed"
HOME_URL = "https://wavespeed.ai"
MODEL_ID = "bytedance/seedream-v4"
SIZE = "1080*1920"
WAIT_TIMEOUT = "4m"
GENERATION_TIMEOUT_S = 300
READ_TIMEOUT_S = 30
INSTALL_LINE = "npx -y @mvanhorn/printing-press-library install wavespeed --cli-only"
KEY_LINE = (
    "Create your own key at https://wavespeed.ai/accesskey and save it: "
    "wavespeed-pp-cli auth set-token <your-key>"
)
SIGNUP_LINE = f"Sign up for WaveSpeed (affiliate link): {WAVESPEED_AFFILIATE_URL}"
FLAGS = ["--json", "--no-input", "--no-color", "--no-learn"]


def redact(text: str) -> str:
    """Drop whole credential-bearing lines."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not any(
            term in line.lower()
            for term in (
                "authorization",
                "bearer",
                "api key",
                "api_key",
                "api-key",
                "apikey",
                "token",
                "secret",
                "password",
                "credential",
                "cookie",
            )
        )
    )


def number(value) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError("expected a finite non-negative number")
    return float(value)


def prediction_id(value) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) else None


def _unwrap(data) -> dict:
    if not isinstance(data, dict):
        raise ValueError("expected an object")
    data = data.get("results", data)
    if not isinstance(data, dict):
        raise ValueError("expected results")
    return data


def setup_lines(state: str) -> list[str]:
    return ([INSTALL_LINE] if state == "missing_cli" else []) + [KEY_LINE]


def unavailable_message(state: str) -> str:
    reason = {
        "missing_cli": "the WaveSpeed CLI is not installed",
        "signed_out": "the WaveSpeed CLI is signed out",
        "offline": "there is no internet connection",
        "rate_limited": "WaveSpeed is rate limited",
    }.get(state, "WaveSpeed could not provide a price")
    message = f"Generated pictures are unavailable: {reason}. Carrying on with your own footage."
    return message + (" " + SIGNUP_LINE if state in {"missing_cli", "signed_out"} else "")


class PicturesUnavailable(YapError):
    """A free read failed before generation."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(
            unavailable_message(reason), error_type="pictures_unavailable", code=reason
        )


class GenerationError(YapError):
    """Keep recovery evidence without retrying a paid request."""

    def __init__(self, message: str, *, identifier=None, price=None, failed=False) -> None:
        self.prediction_id = prediction_id(identifier)
        self.price = price
        self.failed = failed
        super().__init__(redact(message), code="picture_generation_failed")


def safe_detail(text: str, *, prompt="", staging_dir=None) -> str:
    """Keep a short diagnostic without request arguments."""
    private = [prompt, str(staging_dir) if staging_dir else ""]
    return "\n".join(
        line
        for line in redact(text).splitlines()
        if "wavespeed-pp-cli" not in line.lower()
        and not any(value and value in line for value in private)
    )[-1000:]


def _failure_reason(exc, *, error="", prompt="", staging_dir=None) -> str:
    error = error if isinstance(error, str) else ""
    detail = safe_detail(
        error or getattr(exc, "full_stderr", ""), prompt=prompt, staging_dir=staging_dir
    )
    if detail:
        return detail
    code = getattr(exc, "returncode", None)
    return (
        f"WaveSpeed CLI exited with code {code}"
        if type(code) is int
        else "WaveSpeed CLI could not complete the request."
    )


def prediction_failed(prediction: dict) -> bool:
    return prediction.get("status") in {"failed", "error"} or bool(prediction.get("error"))


def status() -> str:
    """Read local auth state without handling a key."""
    if tools.find_tool("wavespeed-pp-cli") is None:
        return "missing_cli"
    try:
        result = tools.run_tool(
            "wavespeed-pp-cli", ["auth", "status", *FLAGS], timeout_s=READ_TIMEOUT_S
        )
        authenticated = _unwrap(json.loads(result.stdout)).get("authenticated")
        return (
            "ready"
            if authenticated is True
            else "signed_out"
            if authenticated is False
            else "unknown"
        )
    except DependencyError:
        return "missing_cli"
    except ProcessingError as exc:
        return "signed_out" if exc.returncode == 4 else "unknown"
    except (YapError, OSError, ValueError, TypeError):
        return "unknown"


def _reason(exc: ProcessingError) -> str:
    lower = redact(exc.full_stderr).lower()
    return (
        "signed_out"
        if exc.returncode == 4
        else "rate_limited"
        if exc.returncode == 7 or "429" in lower or "rate limit" in lower
        else "offline"
        if any(
            term in lower
            for term in (
                "no such host",
                "dial tcp",
                "connection refused",
                "network is unreachable",
                "timeout",
                "tls handshake",
                "network unreachable",
                "name resolution",
            )
        )
        else "price_failed"
    )


def estimate(prompt: str) -> float:
    """Read the current per-image quote without submitting a prediction."""
    try:
        result = tools.run_tool(
            "wavespeed-pp-cli",
            [
                "price",
                "--model-id",
                MODEL_ID,
                f"--prompt={prompt}",
                "--set",
                f"size={SIZE}",
                *FLAGS,
            ],
            timeout_s=READ_TIMEOUT_S,
        )
        return number(_unwrap(json.loads(result.stdout))["data"]["unit_price"])
    except DependencyError:
        raise PicturesUnavailable("missing_cli") from None
    except ProcessingError as exc:
        raise PicturesUnavailable(_reason(exc)) from None
    except (YapError, OSError, ValueError, TypeError, KeyError):
        raise PicturesUnavailable("price_failed") from None


def run_price(data: dict) -> float | None:
    try:
        return number(_unwrap(data["pricing"])["data"]["unit_price"])
    except (ValueError, TypeError, KeyError):
        return None


def generate(prompt: str, staging_dir: Path, *, dry_run: bool):
    """Submit once, or return a redacted free preview."""
    args = [
        "run",
        "--model-id",
        MODEL_ID,
        f"--prompt={prompt}",
        "--set",
        f"size={SIZE}",
        "--price",
        "--wait",
        "--wait-timeout",
        WAIT_TIMEOUT,
        f"--download={Path(staging_dir) / 'out.{ext}'}",
        *FLAGS,
    ]
    if dry_run:
        args.append("--dry-run")
    try:
        result = tools.run_tool("wavespeed-pp-cli", args, timeout_s=GENERATION_TIMEOUT_S)
    except ProcessingError as exc:
        try:
            data = _unwrap(json.loads(getattr(exc, "stdout", "") or "{}"))
        except (ValueError, TypeError):
            data = {}
        prediction = data.get("prediction")
        prediction = prediction.get("data") if isinstance(prediction, dict) else None
        identifier = prediction_id(prediction.get("id")) if isinstance(prediction, dict) else None
        detail = redact(exc.full_stderr)
        match = re.search(r"prediction ([A-Za-z0-9_-]{1,80}) was submitted", detail)
        if identifier is None and match:
            identifier = match[1]
        error = prediction.get("error") if isinstance(prediction, dict) else None
        reason = _failure_reason(exc, error=error or "", prompt=prompt, staging_dir=staging_dir)
        raise GenerationError(
            reason,
            identifier=identifier,
            price=run_price(data),
            failed=isinstance(prediction, dict) and prediction_failed(prediction),
        ) from None
    except (YapError, OSError) as exc:
        raise GenerationError(
            _failure_reason(exc, prompt=prompt, staging_dir=staging_dir)
        ) from None
    if dry_run:
        return redact(result.stderr or "")
    try:
        return _unwrap(json.loads(result.stdout))
    except (ValueError, TypeError):
        raise GenerationError("WaveSpeed returned invalid generation JSON.") from None


def prediction_results(identifier: str) -> dict:
    """Read a prior submission without generating anything."""
    if prediction_id(identifier) is None:
        raise GenerationError("Invalid prediction id.")
    try:
        result = tools.run_tool(
            "wavespeed-pp-cli", ["prediction-results", identifier, *FLAGS], timeout_s=READ_TIMEOUT_S
        )
        data = _unwrap(json.loads(result.stdout))
        prediction = data.get("data")
        if not isinstance(prediction, dict) or prediction.get("id") != identifier:
            raise ValueError("invalid prediction response")
        return data
    except ProcessingError as exc:
        try:
            data = _unwrap(json.loads(getattr(exc, "stdout", "") or "{}"))
        except (ValueError, TypeError):
            data = {}
        prediction = data.get("data")
        if (
            isinstance(prediction, dict)
            and prediction.get("id") == identifier
            and prediction_failed(prediction)
        ):
            return data
        raise GenerationError(_failure_reason(exc), identifier=identifier) from None
    except (YapError, OSError) as exc:
        raise GenerationError(_failure_reason(exc), identifier=identifier) from None
    except (ValueError, TypeError):
        raise GenerationError(
            "WaveSpeed returned invalid prediction JSON.", identifier=identifier
        ) from None


def download(url: str, staging_dir: Path) -> dict:
    """Download a prior output without submitting a paid prediction."""
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or any(ord(c) < 33 for c in url)
        ):
            raise ValueError("invalid download URL")
        _ = parsed.port
    except (ValueError, TypeError, AttributeError):
        raise GenerationError("WaveSpeed returned an invalid output URL.") from None
    try:
        result = tools.run_tool(
            "wavespeed-pp-cli",
            [
                "download",
                url,
                f"--output={staging_dir / 'out.{ext}'}",
                *FLAGS,
            ],
            timeout_s=GENERATION_TIMEOUT_S,
        )
        return _unwrap(json.loads(result.stdout))
    except (YapError, OSError) as exc:
        raise GenerationError(_failure_reason(exc, staging_dir=staging_dir)) from None
    except (ValueError, TypeError):
        raise GenerationError("WaveSpeed returned invalid download JSON.") from None


def charged(identifier: str) -> float | None:
    """Read only a matching charge, discarding all account fields."""
    if prediction_id(identifier) is None:
        return None
    try:
        result = tools.run_tool(
            "wavespeed-pp-cli",
            [
                "billings",
                "--page-size",
                "20",
                "--no-cache",
                "--data-source",
                "live",
                *FLAGS,
            ],
            timeout_s=READ_TIMEOUT_S,
        )
        items = _unwrap(json.loads(result.stdout))["data"]["items"]
        for item in items:
            if item.get("prediction", {}).get("uuid") == identifier:
                return number(item["price"])
    except (YapError, OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return None
