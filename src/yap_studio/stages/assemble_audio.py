"""Mix retained stems on the frame clock and encode the delivery audio once."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg, paths, receipt
from yap_studio.audio_chain import voice_filters
from yap_studio.checks.structure import _loudness
from yap_studio.errors import YapError
from yap_studio.sfx import bleep
from yap_studio.sfx.place import candidates, space
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

RATE = 48_000
SAMPLES_PER_FRAME = 1600
# Positive recipe gain can raise a cue, but its peak must stay this far under voice.
SFX_MAX_RELATIVE_DB = -8.0


class AudioError(YapError):
    """Refuse unusable stems or a delivery stream that fails its audio asserts."""

    def __init__(self, message: str, *, code: str = "assembly_audio_invalid",
                 suggested_action: str | None = None) -> None:
        super().__init__(message, error_type="audio_error", code=code,
                         suggested_action=(suggested_action or
                                           "Correct the audio input or policy and assemble again."))


@dataclass(frozen=True)
class Loudness:
    """Measured integrated loudness, true peak and range; silence has no loudness."""

    integrated_lufs: float | None
    true_peak_dbtp: float | None
    lra_lu: float | None

    def to_dict(self) -> dict[str, float | None]:
        """Return finite JSON-compatible measurements."""
        return asdict(self)


@dataclass(frozen=True)
class AudioResult:
    """The PCM evidence and once-encoded audio for the final receipt."""

    samples: int
    channels: int
    input_voice: Loudness
    pcm_output: Loudness
    policy: dict[str, Any]
    normalization: dict[str, Any] | None
    commands: tuple[tuple[str, ...], ...]
    voice: dict[str, Any]
    music: dict[str, Any]
    sfx: dict[str, Any] | None = None
    click_stem: Path | None = None
    words_stem: Path | None = None
    bleep: dict[str, Any] | None = None
    bleep_voice: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return audio policy and measurements without process objects."""
        return {"policy": self.policy,
                "target": {key: value for key, value in self.policy.items() if key != "mode"}
                if self.policy["mode"] == "loudnorm" else None,
                "input_voice": self.input_voice.to_dict(),
                "pcm_output": self.pcm_output.to_dict(),
                "normalization": self.normalization,
                "samples": self.samples, "channels": self.channels,
                "voice": self.voice, "music": self.music, "sfx": self.sfx,
                **({"bleep": self.bleep} if self.bleep is not None else {})}


def _run(command: list[str], commands: list[tuple[str, ...]], *, binary: bool = False
         ) -> Any:
    """Run a bounded media command and retain its resolved arguments."""
    result = ffmpeg.run(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S, binary=binary)
    commands.append(tuple(result.args))
    return result


def measure(source: Path, commands: list[tuple[str, ...]]) -> Loudness:
    """Measure the actual decoded signal using the verifier's ebur128 parser."""
    result = _run(["ffmpeg", "-nostdin", "-v", "info", "-i", str(source), "-map", "0:a:0",
                   "-af", "ebur128=peak=true", "-f", "null", "-"], commands)
    integrated, lra, peak = _loudness(result.stderr)
    return Loudness(integrated, peak, lra)


def decode(source: Path, commands: list[tuple[str, ...]], *, channels: int) -> np.ndarray:
    """Decode raw samples without masking terminal AAC padding or missing content."""
    result = _run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-map", "0:a:0",
                   "-vn", "-ar", str(RATE), "-ac", str(channels), "-f", "f32le", "-"],
                  commands, binary=True)
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, channels).copy()


def packet_samples(source: Path, commands: list[tuple[str, ...]]) -> list[int]:
    """Read packet durations on the audio stream's native time base."""
    result = _run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                   "stream=time_base:packet=duration", "-of", "json", str(source)], commands)
    value = json.loads(result.stdout)
    scale = Fraction(value["streams"][0]["time_base"]) * RATE
    counts = [Fraction(packet["duration"]) * scale for packet in value["packets"]]
    if any(count.denominator != 1 for count in counts):
        raise AudioError("AAC packet durations do not fall on the 48 kHz sample clock.")
    return [int(count) for count in counts]


def delivery_asserts(source: Path, *, frames: int, policy: dict[str, Any],
                    commands: list[tuple[str, ...]], allow_terminal_padding: bool = False
                    ) -> tuple[list[dict[str, Any]], Loudness]:
    """Check PCM delivery length, packet framing and measured encoded loudness."""
    info = ffmpeg.probe(source)
    expected = frames * SAMPLES_PER_FRAME
    decoded = decode(source, commands, channels=info.channels or 2)
    duration = round((info.audio_duration_s or 0) * RATE)
    packets = packet_samples(source, commands)
    output = measure(source, commands)
    padding = len(decoded) - expected
    checks = [
        {"id": "audio.samples", "pass": duration == expected and
         (0 <= padding < 1024 if allow_terminal_padding else padding == 0),
         "expected": expected, "actual": len(decoded), "container_samples": duration,
         "terminal_padding": padding},
        {"id": "audio.stream", "pass": info.audio_codec == "aac" and info.sample_rate == RATE
         and info.audio_start_s == 0},
        {"id": "audio.aac_packets", "pass": bool(packets) and
         all(count == 1024 for count in packets[:-1]) and 0 < packets[-1] <= 1024,
         "packet_count": len(packets), "last_samples": packets[-1] if packets else None},
    ]
    if policy["mode"] == "loudnorm":
        checks.extend([
            {"id": "audio.integrated", "pass": output.integrated_lufs is not None and
             abs(output.integrated_lufs - policy["integrated_lufs"]) <= 0.5,
             "target": policy["integrated_lufs"], "actual": output.integrated_lufs},
            {"id": "audio.true_peak", "pass": output.true_peak_dbtp is not None and
             output.true_peak_dbtp <= policy["true_peak_dbtp"],
             "target": policy["true_peak_dbtp"], "actual": output.true_peak_dbtp},
        ])
    return checks, output


def _tempo(speed: float) -> str:
    """Factor tempo changes into the supported range without pitch shifting."""
    factors = []
    while speed > 2:
        factors.append(2.0)
        speed /= 2
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    return ",".join(f"atempo={factor:.12f}" for factor in [*factors, speed])


def validate_voice_chain(policy: dict[str, Any], voice: dict[str, Any]) -> str:
    """Refuse voice processing of an approved master before any rendering starts."""
    chain = voice_filters(voice.get("preset", "none"),
                          rumble=voice.get("repair_rumble", False),
                          proximity=voice.get("repair_proximity", "off"),
                          room=voice.get("repair_room", False),
                          harshness=voice.get("repair_harshness", False))
    if chain and policy["mode"] == "preserve":
        raise AudioError("Preserve audio mode refuses a non-empty voice cleanup chain.",
                         code="voice_chain_preserve",
                         suggested_action="Switch the project audio mode to loudnorm or set "
                         "voice preset none and turn off voice repairs.")
    return chain


def _word_ducking(words: list[dict[str, Any]], samples: int, music: dict[str, Any]
                  ) -> tuple[np.ndarray, int]:
    """Merge output speech spans and ramp music gain linearly in decibels.

    Sample ranges are half open. Overlapping ramps use the deeper reduction,
    so adjacent speech never causes double attenuation. File clipping retains
    each ramp's original slope, including speech at the first or last sample.
    """
    spans = []
    try:
        for word in words:
            first, last = float(word["start"]), float(word["end"])
            if not math.isfinite(first) or not math.isfinite(last) or first < 0 or last < first:
                raise ValueError("invalid span")
            start, end = min(samples, round(first * RATE)), min(samples, round(last * RATE))
            if start < end:
                spans.append((start, end))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AudioError(
            "Word ducking needs finite start and end times on the output timeline."
        ) from exc
    gap = round(float(music.get("duck_merge_gap_ms", 250)) * RATE / 1000)
    merged: list[tuple[int, int]] = []
    for first, last in sorted(spans):
        if merged and (first <= merged[-1][1] or first - merged[-1][1] < gap):
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    depth = float(music.get("duck_depth_db", -12))
    attack = round(float(music["duck_attack_ms"]) * RATE / 1000)
    release = round(float(music["duck_release_ms"]) * RATE / 1000)
    decibels = np.zeros(samples, dtype=np.float32)
    for first, last in merged:
        decibels[first:last] = depth
        if attack:
            start = max(0, first - attack)
            ramp = depth * (np.arange(start, first) - (first - attack)) / attack
            decibels[start:first] = np.minimum(decibels[start:first], ramp)
        if release:
            end = min(samples, last + release)
            ramp = depth * (1 - (np.arange(last, end) - last) / release)
            decibels[last:end] = np.minimum(decibels[last:end], ramp)
    return np.power(10.0, decibels / 20).astype(np.float32), len(merged)


def render_audio(project_dir: Path, out: Path, *, work_dir: Path,
                 timeline: ResolvedTimeline, policy: dict[str, Any],
                 music: dict[str, Any], bed: Path | None = None,
                 voice: dict[str, Any] | None = None,
                 cut_words: dict[str, Any] | None = None,
                 bleeps: list[bleep.Occurrence] | None = None,
                 frames: tuple[int, int] | None = None,
                 sfx: dict[str, Any] | None = None,
                 sfx_settings: dict[str, Any] | None = None,
                 flow_look: str = "boxes",
                 specs_rows: str = "together",
                 clips: list[dict[str, Any]] | None = None) -> AudioResult:
    """Keep preserve-mode voice PCM unchanged; mix inserts/music before loudnorm."""
    voice_options = voice or {}
    chain = validate_voice_chain(policy, voice_options)
    duck_mode = music.get("duck_mode", "level")
    if duck_mode not in {"level", "words"}:
        raise AudioError("Choose music duck mode level or words.")
    cut_samples = timeline.total_frames * SAMPLES_PER_FRAME
    samples = (timeline.total_frames + timeline.end_hold_frames) * SAMPLES_PER_FRAME
    ducked_spans = 0
    word_envelope = None
    if bed is not None and duck_mode == "words":
        if (cut_words is None or cut_words.get("time_base") != "output" or
                not isinstance(cut_words.get("words"), list)):
            raise AudioError("Word ducking needs cut words on the output time base.")
        word_envelope, ducked_spans = _word_ducking(cut_words["words"], samples, music)
    commands: list[tuple[str, ...]] = []
    root = Path(project_dir)
    voice_path = paths.validate_input(root / "cut/voice.wav")
    info = ffmpeg.probe(voice_path)
    if info.sample_rate != RATE or not info.channels:
        raise AudioError("cut/voice.wav must contain 48 kHz PCM audio.")
    original = decode(voice_path, commands, channels=info.channels)
    if len(original) != cut_samples or not np.isfinite(original).all():
        raise AudioError("cut/voice.wav has a different sample count from the content picture.")
    input_voice = measure(voice_path, commands)
    audible = [item for item in timeline.elements
               if item.kind == "insert" and item.payload.get("audio", "mute") != "mute"]
    audible.extend(ResolvedElement(
        clip["id"], "card_video", "agent", clip["start_frame"],
        clip["start_frame"] + clip["frames"], "front",
        {"media": {key: clip[key] for key in ("path", "in_s", "speed", "sha256") if key in clip},
         "audio": {"gain_db": clip["gain_db"]}},
    ) for clip in clips or [] if "gain_db" in clip)
    insert_channels = {}
    for item in audible:
        source = paths.validate_input(root / item.payload["media"]["path"])
        if ("sha256" in item.payload["media"]
                and receipt.sha256_file(source) != item.payload["media"]["sha256"]):
            raise AudioError(f"{item.id}: insert media hash changed.")
        source_info = ffmpeg.probe(source, count_frames=False)
        if not source_info.has_audio or not source_info.channels:
            raise AudioError(f"{item.id}: audible insert has no audio stream.")
        insert_channels[item.id] = source_info.channels
    sounds = ([sound for sound in sfx.get("sounds", []) if sound["state"] == "fresh"]
              if sfx and sfx.get("ready") else [])
    if sfx:
        sounds.extend(sound for sound in sfx.get("additional_sounds", [])
                      if sound["state"] == "fresh")
    request_sounds = sfx.get("requested_sounds") if sfx else None
    requests = {sound["name"]: sound for sound in request_sounds or []
                if sound["state"] == "fresh"}
    use_sfx = bool(sounds or requests)
    channels = max(info.channels, 2 if bed or use_sfx or bleeps else 1, *insert_channels.values())
    if channels > 2:
        raise AudioError("Assembly needs mono or stereo stems; convert surround audio first.")
    # Duplicating mono avoids the default upmixer's attenuation of preserved dialogue.
    voice_layout = "pan=stereo|c0=c0|c1=c0," if info.channels == 1 and channels == 2 else ""
    work_dir.mkdir(parents=True, exist_ok=True)
    voice_reference = original
    if (use_sfx or bleeps) and chain:
        treated = work_dir / "sfx-voice-reference.wav"
        _run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(voice_path),
              "-map", "0:a:0", "-af", f"{chain},atrim=end_sample={cut_samples}",
              "-c:a", "pcm_f32le", "-ar", str(RATE), str(treated)], commands)
        voice_reference = decode(treated, commands, channels=info.channels)
        if len(voice_reference) != cut_samples:
            raise AudioError("Voice treatment changed the output sample count.")
    bleep_receipt = None
    bleep_voice = None
    if bleeps:
        muted, tone_stem, bleep_receipt = bleep.render(
            bleeps, voice_reference, cut_samples / RATE, samples)
        bleep_voice = work_dir / "bleep-voice.f32le"
        bleep_voice.write_bytes(muted.astype("<f4").tobytes())
        (work_dir / "bleep.f32le").write_bytes(tone_stem.astype("<f4").tobytes())
    command = ["ffmpeg", "-nostdin", "-v", "error", "-filter_complex_threads", "1"]
    if bleep_voice is not None:
        command.extend(["-f", "f32le", "-ar", str(RATE), "-ac", str(info.channels),
                        "-i", str(bleep_voice)])
    else:
        command.extend(["-i", str(voice_path)])
    treatment = chain + "," if chain else ""
    mix_treatment = "" if bleeps else treatment
    filters = [f"[0:a:0]{voice_layout}{mix_treatment}asetpts=N/SR/TB,apad=whole_len={samples},"
               f"atrim=end_sample={samples}[voice]"]
    index = 1
    mix = ["voice"]
    click_mix = ["voice_stem"] if bed is not None and frames is None else []
    if bleeps:
        clean = (["voice_words"] if frames is None else []) + click_mix
        if bed is not None and duck_mode == "level":
            clean.append("key")
        if clean:
            command.extend(["-i", str(voice_path)])
            filters.append(f"[{index}:a:0]{voice_layout}{treatment}asetpts=N/SR/TB,"
                           f"apad=whole_len={samples},atrim=end_sample={samples}[voice_clean]")
            filters.append("[voice_clean]"
                           + (f"asplit={len(clean)}" if len(clean) > 1 else "anull")
                           + "".join(f"[{label}]" for label in clean))
            index += 1
    if bed is not None:
        bed = paths.validate_input(bed)
        bed_samples = decode(bed, commands, channels=2)
        if len(bed_samples) != samples or np.any(bed_samples[cut_samples:]):
            raise AudioError("music/bed.wav must match the final length with a silent hold.")
        envelope = np.full(samples, float(music["level"]), dtype=np.float32)
        envelope[:round(float(music["dry_open_s"]) * RATE)] = 0
        for item in timeline.elements:
            if item.kind == "music_level":
                envelope[item.start_frame * SAMPLES_PER_FRAME:
                         item.end_frame * SAMPLES_PER_FRAME] *= item.payload["level"]
        if word_envelope is not None:
            envelope *= word_envelope
        raw_bed = work_dir / "leveled-bed.f32le"
        raw_bed.write_bytes((bed_samples * envelope[:, None]).astype("<f4").tobytes())
        command.extend(["-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(raw_bed)])
        if duck_mode == "level":
            if not bleeps:
                filters.append(f"[voice]asplit={3 if click_mix else 2}[voice_mix][key]"
                               + ("[voice_stem]" if click_mix else ""))
            filters.extend([
                f"[{index}:a:0]asetpts=N/SR/TB[bed]",
                f"[bed][key]sidechaincompress=threshold={music['duck_threshold']}:"
                f"ratio={music['duck_ratio']}:attack={music['duck_attack_ms']}:"
                f"release={music['duck_release_ms']}:makeup=1[ducked]",
                f"[ducked]atrim=end_sample={cut_samples},apad=whole_len={samples},"
                f"atrim=end_sample={samples}[music]",
            ])
            mix = ["voice" if bleeps else "voice_mix", "music"]
        else:
            if click_mix and not bleeps:
                filters.append("[voice]asplit=2[voice_mix][voice_stem]")
            filters.append(f"[{index}:a:0]asetpts=N/SR/TB,atrim=end_sample={cut_samples},"
                           f"apad=whole_len={samples},atrim=end_sample={samples}[music]")
            mix = ["voice_mix" if click_mix and not bleeps else "voice", "music"]
        index += 1
    if bleeps:
        command.extend(["-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i",
                        str(work_dir / "bleep.f32le")])
        filters.append(f"[{index}:a:0]asetpts=N/SR/TB[bleep]")
        mix.append("bleep")
        index += 1
    sfx_receipt: dict[str, Any] | None = None
    if sfx is not None:
        sfx_receipt = {"kit": sfx["kit"],
                       "recipes": {item["name"]: item["recipe_sha256"]
                                   for item in sfx.get("sounds", [])},
                       "audio": {item["name"]: item["audio_sha256"]
                                 for item in sfx.get("sounds", []) if item["state"] == "fresh"},
                       "cues": [], "dropped": []}
        if "fallback_from" in sfx:
            sfx_receipt["fallback_from"] = sfx["fallback_from"]
        if "account" in sfx:
            sfx_receipt["account"] = sfx["account"]
        if request_sounds is not None:
            sfx_receipt["requests"] = {sound["name"]: {
                "recipe_sha256": sound["recipe_sha256"],
                **({"audio_sha256": sound["audio_sha256"]} if sound["state"] == "fresh" else {}),
            } for sound in request_sounds}
    if use_sfx and sfx is not None and sfx_receipt is not None:
        settings = sfx_settings or {}
        gap = float(settings.get("sfx_min_gap_s", 2.5))
        below = float(settings.get("sfx_peak_below_voice_db", 10))
        sfx_receipt["recipes"] = {sound["name"]: sound["recipe_sha256"] for sound in sounds}
        sfx_receipt["audio"] = {sound["name"]: sound["audio_sha256"] for sound in sounds}
        by_name = {sound["name"]: sound for sound in sounds}
        selected, dropped = space(candidates(timeline, sounds, flow_look=flow_look,
                                             specs_rows=specs_rows,
                                             sound_extras=sfx.get("sound_extras", []),
                                             requested_sounds=request_sounds), gap)
        sfx_receipt["dropped"] = dropped
        stem = np.zeros((samples, 2), dtype=np.float32)
        whole_voice_peak = float(np.max(np.abs(voice_reference)))
        decoded: dict[str, np.ndarray] = {}
        cue_windows: list[tuple[dict[str, Any], int, int]] = []
        for cue in selected:
            sound = (requests if cue.rule == "request" and request_sounds is not None
                     else by_name)[cue.sound]
            if sound.get("made"):
                from yap_studio.sfx import made

                audio = made.render(cue.sound)
                if hashlib.sha256(audio.tobytes()).hexdigest() != sound["audio_sha256"]:
                    raise AudioError(f"{cue.sound}: made sound changed during assembly.")
            else:
                source = paths.validate_input(Path(sound["cache_path"]))
                if receipt.sha256_file(source) != sound["audio_sha256"]:
                    raise AudioError(f"{cue.sound}: cached sound changed during assembly.")
                if str(source) not in decoded:
                    decoded[str(source)] = decode(source, commands, channels=2)
                audio = decoded[str(source)]
            if cue.align_peak and "peak_s" not in sound:
                cue = replace(cue, peak_s=int(np.argmax(np.max(np.abs(audio), axis=1))) / RATE)
            start = round((cue.time_s - cue.peak_s) * RATE)
            if cue.event == "end":
                start = max(0, min(start, cut_samples - len(audio) - round(0.05 * RATE)))
            head = max(0, -start)
            start = max(0, start)
            end = min(cut_samples, start + len(audio) - head)
            if cue.end_s is not None:
                end = min(end, round(cue.end_s * RATE))
            if end <= start:
                continue
            sample = audio[head:head + end - start]
            trim = None
            trimmed = cue.end_s is not None and len(sample) < len(audio) - head
            fade_out_s = cue.fade_out_s or (0.06 if trimmed else 0)
            fade_samples = min(len(sample), round(fade_out_s * RATE))
            if fade_samples:
                sample = sample.copy()
                sample[-fade_samples:] *= np.linspace(1, 0, fade_samples)[:, None]
            if trimmed:
                trim = {"duration_s": len(sample) / RATE, "fade_out_s": fade_samples / RATE}
            source_peak = float(np.max(np.abs(sample)))
            if source_peak <= 0:
                continue
            # The local spoken peak drives a cue. Silent windows use the whole take.
            near = voice_reference[max(0, start - RATE // 4):min(cut_samples, end + RATE // 4)]
            voice_peak = float(np.max(np.abs(near))) if len(near) else 0.0
            voice_peak = voice_peak or whole_voice_peak
            if voice_peak <= 0:
                continue
            target_peak = voice_peak * 10 ** ((-below + sound["gain_db"] + cue.gain_db) / 20)
            target_peak = min(target_peak, voice_peak * 10 ** (SFX_MAX_RELATIVE_DB / 20))
            gain = target_peak / source_peak
            stem[start:end] += sample * gain
            record = {**cue.to_dict(), "time_s": start / RATE,
                      "gain_db": 20 * math.log10(gain),
                      "peak_dbfs": 20 * math.log10(target_peak),
                      "voice_peak_dbfs": 20 * math.log10(voice_peak),
                      "relative_db": 20 * math.log10(target_peak / voice_peak)}
            if trim is not None:
                record["trim"] = trim
            elif fade_samples:
                record["fade_out_s"] = fade_samples / RATE
            if sound.get("made") or "peak_s" in sound or cue.rule == "request":
                record["peak_s"] = cue.peak_s
            peak_index = int(np.argmax(np.max(np.abs(sample), axis=1)))
            record["peak_at_s"] = (start + peak_index) / RATE
            sfx_receipt["cues"].append(record)
            cue_windows.append((record, start, end))
        if np.any(stem[cut_samples:]):
            raise AudioError("Sound effects entered the silent end hold.")
        ratios = [float(np.max(np.abs(stem[start:end]))) /
                  (10 ** (record["voice_peak_dbfs"] / 20))
                  for record, start, end in cue_windows]
        maximum = max(ratios, default=0)
        if maximum > 10 ** (SFX_MAX_RELATIVE_DB / 20):
            correction = 10 ** (SFX_MAX_RELATIVE_DB / 20) / maximum
            stem *= correction
            for record, _, _ in cue_windows:
                record["gain_db"] += 20 * math.log10(correction)
        for record, start, end in cue_windows:
            mixed_peak = float(np.max(np.abs(stem[start:end])))
            if mixed_peak > 0:
                record["peak_dbfs"] = 20 * math.log10(mixed_peak)
                record["relative_db"] = record["peak_dbfs"] - record["voice_peak_dbfs"]
            if record["relative_db"] >= 0:
                raise AudioError("A sound cue exceeded the voice peak.")
        stem_path = work_dir / "sfx.f32le"
        stem_path.write_bytes(stem.astype("<f4").tobytes())
        command.extend(["-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(stem_path)])
        filters.append(f"[{index}:a:0]asetpts=N/SR/TB,atrim=end_sample={cut_samples},"
                       f"apad=whole_len={samples},atrim=end_sample={samples}[sfx]")
        mix.append("sfx")
        index += 1
    for item in audible:
        gain = item.payload["audio"]
        media = item.payload["media"]
        source = paths.validate_input(root / media["path"])
        command.extend(["-i", str(source)])
        count = (item.end_frame - item.start_frame) * SAMPLES_PER_FRAME
        label = f"insert{index}"
        layout = ("pan=stereo|c0=c0|c1=c0,"
                  if insert_channels[item.id] == 1 and channels == 2 else "")
        insert_filter = (
            f"[{index}:a:0]{layout}aresample={RATE},atrim=start={media.get('in_s', 0):.12f},"
            f"asetpts=N/SR/TB,{_tempo(float(media.get('speed', 1)))},"
            f"apad=whole_len={count},atrim=end_sample={count},volume={gain['gain_db']}dB,"
            f"adelay={item.start_frame * SAMPLES_PER_FRAME}S:all=1,"
            f"apad=whole_len={samples},atrim=end_sample={samples}"
        )
        if click_mix:
            filters.append(f"{insert_filter}[{label}_split];"
                           f"[{label}_split]asplit=2[{label}][{label}_stem]")
            click_mix.append(f"{label}_stem")
        else:
            filters.append(f"{insert_filter}[{label}]")
        mix.append(label)
        index += 1
    words_mix = ["voice_words"] if frames is None and bleeps else []
    if frames is None and (bleeps or (sfx_receipt is not None and sfx_receipt["cues"])):
        for position, label in enumerate(mix):
            if label not in ({"sfx", "bleep", "voice"} if bleeps else {"sfx"}):
                filters.append(f"[{label}]asplit=2[{label}_final][{label}_words]")
                mix[position] = f"{label}_final"
                words_mix.append(f"{label}_words")
    joined = "".join(f"[{label}]" for label in mix)
    expression = (f"amix=inputs={len(mix)}:normalize=0:dropout_transition=0"
                  if len(mix) > 1 else "anull")
    filters.append(f"{joined}{expression},atrim=end_sample={cut_samples},"
                   f"apad=whole_len={samples},atrim=end_sample={samples},asetpts=N/SR/TB[mix]")
    if click_mix:
        click_joined = "".join(f"[{label}]" for label in click_mix)
        click_expression = (f"amix=inputs={len(click_mix)}:normalize=0:dropout_transition=0"
                            if len(click_mix) > 1 else "anull")
        filters.append(f"{click_joined}{click_expression},atrim=end_sample={cut_samples},"
                       f"apad=whole_len={samples},atrim=end_sample={samples},"
                       "asetpts=N/SR/TB[click]")
    if words_mix:
        words_joined = "".join(f"[{label}]" for label in words_mix)
        words_expression = (f"amix=inputs={len(words_mix)}:normalize=0:dropout_transition=0"
                            if len(words_mix) > 1 else "anull")
        filters.append(f"{words_joined}{words_expression},atrim=end_sample={cut_samples},"
                       f"apad=whole_len={samples},atrim=end_sample={samples},"
                       "asetpts=N/SR/TB[words]")
    mixed = work_dir / "mix.wav"
    click_stem = work_dir / "voice-stem.wav" if click_mix else None
    words_stem = work_dir / "no-effects-stem.wav" if words_mix else None
    command.extend(["-filter_complex", ";".join(filters), "-map", "[mix]", "-c:a", "pcm_f32le",
                    "-ar", str(RATE), str(mixed)])
    if click_stem is not None:
        command.extend(["-map", "[click]", "-c:a", "pcm_f32le", "-ar", str(RATE),
                        str(click_stem)])
    if words_stem is not None:
        command.extend(["-map", "[words]", "-c:a", "pcm_f32le", "-ar", str(RATE),
                        str(words_stem)])
    _run(command, commands)
    pcm = mixed
    normalization = None
    if policy["mode"] == "loudnorm":
        # AAC reconstruction needs headroom below the requested delivery true peak.
        peak = max(-9.0, float(policy["true_peak_dbtp"]) - 0.5)
        norm = (f"loudnorm=I={policy['integrated_lufs']}:TP={peak}:LRA={policy['lra_lu']}")
        measured = _run(["ffmpeg", "-nostdin", "-v", "info", "-i", str(mixed), "-map", "0:a:0",
                         "-af", norm + ":print_format=json", "-f", "null", "-"], commands)
        try:
            stats, _ = json.JSONDecoder().raw_decode(measured.stderr[measured.stderr.rfind("{"):])
            if not all(math.isfinite(float(stats[key])) for key in
                       ("input_i", "input_tp", "input_lra", "input_thresh")):
                raise ValueError("unmeasurable signal")
        except (ValueError, KeyError, TypeError) as exc:
            raise AudioError("The mix has no measurable loudness for normalisation.") from exc
        # ffmpeg's linear=true falls back to dynamic normalisation when its true-peak target
        # is impossible. Apply this measured integrated-loudness offset directly so dialogue
        # keeps one gain curve; the oversampled limiter alone handles true peaks.
        gain_db = float(policy["integrated_lufs"]) - float(stats["input_i"])
        limit = 10 ** (peak / 20)
        # A short look-ahead limiter may slightly lower integrated loudness. Re-measure and
        # correct the one fixed gain a few times instead of asking loudnorm to go dynamic.
        for attempt in range(3):
            applied_gain_db = gain_db
            second = (f"volume={applied_gain_db:.12f}dB,aresample=192000,"
                      f"alimiter=limit={limit:.12f}:attack=0.1:release=1:level=false:"
                      f"latency=true,aresample={RATE},asetpts=N/SR/TB,"
                      f"atrim=end_sample={cut_samples},apad=whole_len={samples},"
                      f"atrim=end_sample={samples}")
            pcm = work_dir / f"normalised-{attempt}.wav"
            _run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(mixed), "-map", "0:a:0",
                  "-af", second, "-c:a", "pcm_f32le", str(pcm)], commands)
            applied = measure(pcm, commands)
            if applied.integrated_lufs is None:
                raise AudioError("The limited mix has no measurable loudness.")
            correction = float(policy["integrated_lufs"]) - applied.integrated_lufs
            if abs(correction) <= 0.05 or attempt == 2:
                break
            gain_db += correction
        normalization = {"mode": "constant_gain_limiter", "gain_db": applied_gain_db,
                         "limiter_dbtp": peak, "iterations": attempt + 1}
        for raw_stem in (click_stem, words_stem):
            if raw_stem is None:
                continue
            normalised_stem = raw_stem.with_stem(raw_stem.stem + "-normalised")
            _run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(raw_stem),
                  "-map", "0:a:0", "-af", second, "-c:a", "pcm_f32le", str(normalised_stem)],
                 commands)
            normalised_stem.replace(raw_stem)
    final_pcm = decode(pcm, commands, channels=channels)
    if len(final_pcm) != samples or np.any(final_pcm[cut_samples:]):
        raise AudioError("The PCM mix failed its exact length or silent hold assert.")
    if words_stem is not None:
        words_pcm = decode(words_stem, commands, channels=channels)
        if len(words_pcm) != samples or np.any(words_pcm[cut_samples:]):
            raise AudioError("The no-effects stem failed its exact length or silent hold assert.")
    if policy["mode"] == "preserve" and len(mix) == 1:
        if not np.array_equal(final_pcm[:cut_samples], original):
            raise AudioError("Preserve mode changed the retained voice samples.")
    measured_pcm = pcm
    if frames is not None:
        first, last = frames
        final_pcm = final_pcm[first * SAMPLES_PER_FRAME:(last + 1) * SAMPLES_PER_FRAME]
        pcm = work_dir / "pilot.f32le"
        pcm.write_bytes(final_pcm.astype("<f4", copy=False).tobytes())
    target = paths.validate_output(out, allowed_existing_suffixes=frozenset())
    source_args = (["-f", "f32le", "-ar", str(RATE), "-ac", str(channels)]
                   if frames is not None else [])
    _run(["ffmpeg", "-nostdin", "-v", "error", *source_args, "-i", str(pcm),
          "-map", "0:a:0", "-c:a", "aac", "-b:a", "256k", "-ar", str(RATE),
          "-movie_timescale", str(RATE), str(target)], commands)
    proximity = voice_options.get("repair_proximity", "off")
    proximity_name = "proximity_strong" if proximity == "strong" else "proximity"
    repairs = [name for name, enabled in (
        ("rumble", voice_options.get("repair_rumble", False)),
        (proximity_name, proximity != "off"),
        ("room", voice_options.get("repair_room", False)),
        ("harshness", voice_options.get("repair_harshness", False)),
    ) if enabled]
    return AudioResult(len(final_pcm), final_pcm.shape[1], input_voice,
                       measure(measured_pcm, commands),
                       dict(policy), normalization, tuple(commands),
                       {"preset": voice_options.get("preset", "none"),
                        "repairs": repairs,
                        "filters": chain},
                       {"duck_mode": duck_mode,
                        **({"ducked_spans": ducked_spans} if duck_mode == "words" else {})},
                       sfx_receipt, click_stem, words_stem, bleep_receipt, bleep_voice)
