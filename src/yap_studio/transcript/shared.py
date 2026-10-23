"""Content-addressed raw transcripts shared by local speech consumers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from yap_studio import cache, receipt
from yap_studio.errors import InputError, YapError
from yap_studio.project.receipts import _identity

from .backends import ENGINES, Backend
from .model import Transcript

SCHEMA = "yap-studio/transcript-cache@1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Request:
    """Identify a take's selected audio and the engine's input media."""

    take_sha256: str
    audio_stream: int
    media: Path
    source: Path | None = None
    source_entry: dict | None = field(default=None, compare=False, hash=False)


def _model_id(model: str) -> str:
    local = Path(model)
    return str(local.resolve()) if local.is_dir() or local.is_file() else model


def model_spec(backend: Backend) -> str:
    """Return the model argument that rebuilds the same backend through resolve_backend."""
    model = _model_id(backend.model)
    if backend.name == "parakeet-onnx" and getattr(backend, "quantization", None) is None:
        return f"{model}:full"
    return model


def _key(backend: Backend, request: Request, language: str | None) -> dict:
    if _SHA256.fullmatch(request.take_sha256) is None:
        raise ValueError("take sha256 must be 64 lowercase hex characters")
    if type(request.audio_stream) is not int or request.audio_stream < 0:
        raise ValueError("audio stream must be a non-negative ordinal")
    version = backend.version()
    return {"take_sha256": request.take_sha256, "engine": backend.name,
            "model": _model_id(backend.model), "version": "unknown" if version is None else version,
            "quantization": (getattr(backend, "quantization", None)
                             if backend.name == "parakeet-onnx" else None),
            "compute_type": (getattr(backend, "compute_type", None)
                             if backend.name == "faster-whisper" else None),
            "audio_stream": request.audio_stream,
            "language": language if backend.name == "faster-whisper" else None}


def _entry(key: dict) -> Path:
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"))
    variant = hashlib.sha256(canonical.encode()).hexdigest()[:24]
    return cache.cache_dir("transcripts", key["take_sha256"]) / f"{variant}.json"


def _read(entry: Path, key: dict) -> Transcript | None:
    try:
        data = json.loads(entry.read_bytes())
        if (not isinstance(data, dict) or data.get("schema") != SCHEMA
                or any(name not in data or type(data[name]) is not type(value)
                       or data[name] != value
                       for name, value in key.items())):
            return None
        return Transcript.from_dict(data["transcript"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError, YapError, RecursionError):
        return None


def lookup(backend: Backend, request: Request, *, language: str | None = None
           ) -> Transcript | None:
    """Read a matching raw transcript without running a speech engine."""
    if backend.name not in ENGINES:
        return None
    key = _key(backend, request, language)
    try:
        result = _read(_entry(key), key)
        if result is not None:
            return result
        if backend.name == "faster-whisper" and language is not None:
            detected_key = {**key, "language": None}
            result = _read(_entry(detected_key), detected_key)
            if result is not None and result.language == language:
                return result
    except OSError:
        pass
    return None


def _save(target: Path, data: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.stem}-", suffix=".tmp",
                                         delete=False) as output:
            temporary = Path(output.name)
            json.dump(data, output, separators=(",", ":"))
        os.replace(temporary, target)
    except OSError:
        pass  # Raw speech results can always be made again.
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def store(backend: Backend, request: Request, result: Transcript, *, language: str | None = None
          ) -> None:
    """Save engine output before a consumer changes words or timestamps."""
    if backend.name not in ENGINES:
        return
    key = _key(backend, request, language)
    try:
        _save(_entry(key), {"schema": SCHEMA, **key, "transcript": result.to_dict()})
    except OSError:
        pass


def _request_identities(requests: Sequence[Request]) -> dict[Path, dict]:
    identities = {}
    for request in requests:
        source = request.source or request.media
        try:
            identity = _identity(source)
            entry = request.source_entry
            if entry is None:
                try:
                    entry = json.loads(_memo(source.resolve()).read_bytes())
                except (OSError, ValueError, TypeError):
                    pass
            if isinstance(entry, dict) and entry.get("sha256") == request.take_sha256:
                matches = all(entry.get(name) == value for name, value in identity.items())
            else:
                matches = receipt.sha256_file(source) == request.take_sha256
            if not matches:
                raise InputError(str(source), "file changed since hashing; re-read and retry")
            identities.setdefault(source, identity)
            identities.setdefault(request.media, _identity(request.media))
        except OSError as exc:
            raise InputError(str(source), "file changed or disappeared; re-read and retry") from exc
    return identities


def _check_identities(identities: dict[Path, dict]) -> None:
    for path, identity in identities.items():
        try:
            if _identity(path) == identity:
                continue
        except OSError:
            pass
        raise InputError(str(path), "file changed or disappeared; re-read and retry")


def transcribe(backend: Backend, requests: Sequence[Request], *, language: str | None = None,
               refresh: bool = False) -> dict[Path, tuple[Transcript, bool]]:
    """Batch misses under ordered per-entry locks, then publish raw engine output."""
    if not requests:
        return {}
    if backend.name not in ENGINES:
        return {path: (result, False) for path, result in backend.transcribe_many(
            [request.media for request in requests], language=language,
        ).items()}
    keys = {request: _key(backend, request, language) for request in requests}
    identities = _request_identities(requests)
    results = {}
    missing = []
    for request in requests:
        hit = None if refresh else lookup(backend, request, language=language)
        if hit is None:
            missing.append(request)
        else:
            results[request.media] = (hit, True)
    with ExitStack() as locks:
        entries = {}
        for request in missing:
            try:
                entries[request] = _entry(keys[request])
            except OSError:
                pass
        for entry in sorted(set(entries.values())):
            try:
                handle = locks.enter_context(entry.with_suffix(".lock").open("a"))
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(handle, fcntl.LOCK_EX)
            except OSError:
                pass
        pending = {}
        aliases = {}
        for request in missing:
            hit = None if refresh else lookup(backend, request, language=language)
            if hit is not None:
                results[request.media] = (hit, True)
                continue
            identifier = entries.get(request, request)
            if identifier in pending:
                aliases[request] = pending[identifier]
            else:
                pending[identifier] = request
        if pending:
            raw = backend.transcribe_many([item.media for item in pending.values()],
                                          language=language)
            _check_identities(identities)
            for request in pending.values():
                if request.media in raw:
                    result = raw[request.media]
                    store(backend, request, result, language=language)
                    results[request.media] = (result, False)
            for alias, request in aliases.items():
                if request.media in raw:
                    result = Transcript.from_dict(raw[request.media].to_dict())
                    results[alias.media] = (result, False)
    _check_identities(identities)
    return results


def _memo(path: Path) -> Path:
    name = hashlib.sha256(str(path).encode()).hexdigest()
    return cache.cache_dir("transcripts", "paths") / f"{name}.json"


def remember_digest(path: Path, sha256: str, identity: dict) -> None:
    """Remember a hash with its recorded identity, without taking a fresh stat."""
    if _SHA256.fullmatch(sha256) is None:
        raise ValueError("take sha256 must be 64 lowercase hex characters")
    try:
        _save(_memo(path.resolve()), {**identity, "sha256": sha256})
    except OSError:
        pass


def known_digest(path: Path) -> str | None:
    """Read a memo only when the current file identity matches, without hashing."""
    try:
        real = path.resolve()
        identity = _identity(real)
        data = json.loads(_memo(real).read_bytes())
        if (isinstance(data, dict)
                and all(data.get(name) == value for name, value in identity.items())
                and isinstance(data.get("sha256"), str)
                and _SHA256.fullmatch(data["sha256"])):
            return data["sha256"]
    except (OSError, ValueError, TypeError):
        pass
    return None


def take_digest(path: Path) -> str:
    """Hash a stable take once and remember its size, times, inode and device."""
    known = known_digest(path)
    if known is not None:
        return known
    real = path.resolve()
    try:
        identity = _identity(real)
        for _ in range(3):
            digest = receipt.sha256_file(real)
            after = _identity(real)
            if identity == after:
                remember_digest(real, digest, identity)
                return digest
            identity = after
    except OSError as exc:
        raise InputError(str(path), "file changed or disappeared; re-read and retry") from exc
    raise InputError(str(path), "file kept changing while hashing; wait and retry")
