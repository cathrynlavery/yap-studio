"""Small deterministic sounds made from oscillators, noise and envelopes."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio.sfx import _NAME, RULES, SfxError

RATE = 48000
SYNTH_VERSION = 1


def _fields(value: Any, required: set[str], optional: set[str], where: str) -> None:
    if not isinstance(value, dict):
        raise SfxError(f"{where}: expected an object.")
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing or extra:
        raise SfxError(f"{where}: invalid fields {', '.join(sorted(missing | extra))}.")


def _number(value: Any, low: float, high: float, where: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise SfxError(f"{where}: use a number from {low} to {high}.")


def _sweep(value: dict, where: str) -> None:
    for field in ("from_hz", "to_hz", "via_hz"):
        if field in value:
            _number(value[field], 10, 20000, f"{where}.{field}")


def _check_sound(sound: Any, where: str) -> dict:
    _fields(sound, {"name", "rule", "priority", "gain_db", "duration_s", "peak_s",
                    "seed", "layers"}, set(), where)
    if not isinstance(sound["name"], str) or not _NAME.fullmatch(sound["name"]):
        raise SfxError(f"{where}.name: use lowercase letters, digits and hyphens.")
    if not isinstance(sound["rule"], str) or sound["rule"] not in RULES:
        raise SfxError(f"{where}.rule: choose {', '.join(sorted(RULES))}.")
    for field, low, high in (("duration_s", .01, 10), ("peak_s", 0, sound["duration_s"]),
                             ("gain_db", -40, 12)):
        _number(sound[field], low, high, f"{where}.{field}")
    if round(sound["peak_s"] * RATE) >= round(sound["duration_s"] * RATE):
        raise SfxError(f"{where}.peak_s: must be before duration_s.")
    for field, low, high in (("priority", 1, 100), ("seed", 0, 2**32 - 1)):
        if type(sound[field]) is not int or not low <= sound[field] <= high:
            raise SfxError(f"{where}.{field}: use an integer from {low} to {high}.")
    if not isinstance(sound["layers"], list) or not 1 <= len(sound["layers"]) <= 16:
        raise SfxError(f"{where}.layers: use 1 to 16 layers.")
    for index, layer in enumerate(sound["layers"]):
        loc = f"{where}.layers[{index}]"
        _fields(layer, {"source", "level", "envelope"},
                {"from_hz", "to_hz", "via_hz", "filter", "repeat"}, loc)
        if layer["source"] not in ("noise", "sine", "triangle", "square"):
            raise SfxError(f"{loc}.source: choose noise, sine, triangle or square.")
        _number(layer["level"], 0, 1, f"{loc}.level")
        _sweep(layer, loc)
        if layer["source"] != "noise" and "from_hz" not in layer:
            raise SfxError(f"{loc}.from_hz: an oscillator needs a pitch.")
        env = layer["envelope"]
        _fields(env, {"start_s", "attack_s", "hold_s", "release_s"},
                {"attack_curve", "release_curve", "decay"}, f"{loc}.envelope")
        for field in ("start_s", "attack_s", "hold_s", "release_s"):
            _number(env[field], 0, sound["duration_s"], f"{loc}.envelope.{field}")
        if env["attack_s"] + env["release_s"] <= 0:
            raise SfxError(f"{loc}.envelope.release_s: envelope must have a duration.")
        for field in ("attack_curve", "release_curve", "decay"):
            if field in env:
                _number(env[field], .1, 30, f"{loc}.envelope.{field}")
        end = sum(env[field] for field in ("start_s", "attack_s", "hold_s", "release_s"))
        if "repeat" in layer:
            repeat = layer["repeat"]
            _fields(repeat, {"count", "every_s", "decay"}, set(), f"{loc}.repeat")
            if type(repeat["count"]) is not int or not 1 <= repeat["count"] <= 16:
                raise SfxError(f"{loc}.repeat.count: use an integer from 1 to 16.")
            _number(repeat["every_s"], .001, 10, f"{loc}.repeat.every_s")
            _number(repeat["decay"], 0, 1, f"{loc}.repeat.decay")
            end += (repeat["count"] - 1) * repeat["every_s"]
        if end > sound["duration_s"] + 1 / RATE:
            raise SfxError(f"{loc}.envelope: envelope and repeat exceed duration_s.")
        if "filter" in layer:
            filt = layer["filter"]
            _fields(filt, {"type", "from_hz", "to_hz"}, {"via_hz", "q"}, f"{loc}.filter")
            if filt["type"] not in ("lowpass", "highpass", "bandpass"):
                raise SfxError(f"{loc}.filter.type: choose lowpass, highpass or bandpass.")
            _sweep(filt, f"{loc}.filter")
            if "q" in filt:
                _number(filt["q"], .2, 8, f"{loc}.filter.q")
    return sound


@lru_cache(maxsize=1)
def _kit() -> dict:
    data = json.loads((Path(__file__).parent / "kits/made.json").read_text())
    _fields(data, {"schema", "name", "label", "description", "sounds"}, set(), "made")
    if data["schema"] != "yap-studio/sfx-made-kit@1" or data["name"] != "made":
        raise SfxError("made.schema: expected yap-studio/sfx-made-kit@1 with name made.")
    for field in ("label", "description"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise SfxError(f"made.{field}: expected text.")
    if not isinstance(data["sounds"], list) or not data["sounds"]:
        raise SfxError("made.sounds: expected recipes.")
    names = set()
    for index, sound in enumerate(data["sounds"]):
        _check_sound(sound, f"made.sounds[{index}]")
        if sound["name"] in names:
            raise SfxError(f"made.sounds[{index}].name: duplicate sound.")
        names.add(sound["name"])
    return data


def kit() -> dict:
    """Return the validated recipes without sharing mutable cache data."""
    return copy.deepcopy(_kit())


def _noise(count: int, seed: int) -> np.ndarray:
    # Counter hashing uses fixed integer operations, independent of random distributions.
    bits = np.arange(count, dtype=np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    bits = (bits ^ (bits >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    bits = (bits ^ (bits >> 27)) * np.uint64(0x94D049BB133111EB)
    bits ^= bits >> 31
    return (bits >> 11).astype(np.float64) * (2 / 2**53) - 1


def _frequency(spec: dict, count: int) -> np.ndarray:
    first = spec["from_hz"]
    last = spec.get("to_hz", first)
    position = np.linspace(0, 1, count)
    if "via_hz" in spec:
        middle = spec["via_hz"]
        return np.exp(np.interp(position, [0, .5, 1], np.log([first, middle, last])))
    return first * np.exp(position * math.log(last / first))


def _filter(audio: np.ndarray, spec: dict) -> np.ndarray:
    cutoff = _frequency(spec, len(audio))
    result = np.empty_like(audio)
    if spec["type"] in ("lowpass", "highpass"):
        # y[n] = (1-a[n])x[n] + a[n]y[n-1]; highpass is x[n]-y[n].
        poles = np.exp(-2 * np.pi * cutoff / RATE)
        previous = 0.0
        for index, (value, pole) in enumerate(zip(audio.tolist(), poles.tolist(), strict=True)):
            previous = (1 - pole) * value + pole * previous
            result[index] = value - previous if spec["type"] == "highpass" else previous
        return result
    # Bandpass biquad from a damped second-order resonator, with unit centre gain.
    omega = 2 * np.pi * cutoff / RATE
    alpha = np.sin(omega) / (2 * spec.get("q", .7))
    b = alpha / (1 + alpha)
    a1 = -2 * np.cos(omega) / (1 + alpha)
    a2 = (1 - alpha) / (1 + alpha)
    x1 = x2 = y1 = y2 = 0.0
    for index, (value, gain, pole1, pole2) in enumerate(zip(
            audio.tolist(), b.tolist(), a1.tolist(), a2.tolist(), strict=True)):
        output = gain * (value - x2) - pole1 * y1 - pole2 * y2
        result[index] = output
        x2, x1, y2, y1 = x1, value, y1, output
    return result


def _envelope(time: np.ndarray, env: dict) -> np.ndarray:
    local = time - env["start_s"]
    attack, hold, release = (env[field] for field in ("attack_s", "hold_s", "release_s"))
    rising = np.clip(local / max(attack, 1 / RATE), 0, 1) ** env.get("attack_curve", 1)
    remaining = np.clip(1 - (local - attack - hold) / max(release, 1 / RATE), 0, 1)
    falling = remaining ** env.get("release_curve", 1)
    if "decay" in env:
        decay = env["decay"]
        falling = (np.exp(-decay * (1 - remaining)) - math.exp(-decay)) / (
            1 - math.exp(-decay))
    return np.where(local >= 0, rising * falling, 0)


@lru_cache(maxsize=16)
def render(name: str) -> np.ndarray:
    """Render immutable stereo float32 PCM at 48 kHz."""
    sound = next((item for item in _kit()["sounds"] if item["name"] == name), None)
    if sound is None:
        raise SfxError(f"Unknown made sound name: {name}.")
    count = round(sound["duration_s"] * RATE)
    time = np.arange(count) / RATE
    audio = np.zeros(count)
    for index, layer in enumerate(sound["layers"]):
        if layer["source"] == "noise":
            source = _noise(count, sound["seed"] + (index + 1) * 1000003)
        else:
            phase = np.cumsum(_frequency(layer, count)) / RATE
            cycle = phase % 1
            source = (np.sin(2 * np.pi * phase) if layer["source"] == "sine" else
                      1 - 4 * np.abs(cycle - .5) if layer["source"] == "triangle" else
                      np.where(cycle < .5, 1.0, -1.0))
        if "filter" in layer:
            source = _filter(source, layer["filter"])
        repeat = layer.get("repeat", {"count": 1, "every_s": 0, "decay": 1})
        envelope = sum(repeat["decay"] ** tick * _envelope(
            time - tick * repeat["every_s"], layer["envelope"])
            for tick in range(repeat["count"]))
        audio += source * envelope * layer["level"]
    audio *= .89 / np.max(np.abs(audio))
    stereo = np.column_stack((audio, audio)).astype("<f4")
    stereo.setflags(write=False)
    return stereo


def wav_bytes(name: str) -> bytes:
    """Encode a deterministic stereo 16-bit PCM WAV in memory."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        output.writeframes(np.rint(render(name) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def peak_time(name: str) -> float:
    """Measure the loudest sample on the sound's own clock."""
    return int(np.argmax(np.abs(render(name)[:, 0]))) / RATE


def recipe_sha256(sound: dict) -> str:
    """Bind the recipe and the synthesis version."""
    return hashlib.sha256(json.dumps({"version": SYNTH_VERSION, "sound": sound},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audio_sha256(name: str) -> str:
    """Identify the rendered float32 PCM."""
    return hashlib.sha256(render(name).tobytes()).hexdigest()


def listing() -> dict:
    """Describe the ready kit without writing audio to disk."""
    data = _kit()
    return {"kit": "made", "label": data["label"], "description": data["description"],
            "sounds": [{**{key: sound[key] for key in (
                "name", "rule", "priority", "gain_db", "duration_s")},
                "peak_s": peak_time(sound["name"]),
                "made": True, "state": "fresh", "cache_path": None,
                "recipe_sha256": recipe_sha256(sound),
                "audio_sha256": audio_sha256(sound["name"])} for sound in data["sounds"]]}
