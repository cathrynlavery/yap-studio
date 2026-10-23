"""Fixed speech treatments at 48 kHz, with unity makeup gain and no timing shift."""

from __future__ import annotations

from yap_studio.errors import YapError

# Each entry is a complete, fixed treatment. The two-pole highpass cuts rumble;
# denoising is limited to 6 dB; compression uses no lookahead and no makeup gain.
# afftdn delays audio by two 12.5 ms blocks at 48 kHz. Feed 1200 trailing samples
# before removing that delay, so the final input samples are retained as well.
# The room repair uses a mild downward expander, capped at about 6 dB reduction.
# It lowers quiet room tails without trying to remove reverberation from speech.
# Proximity reduces boomy bass from a close mic.
# Its 100 Hz highpass replaces the 80 Hz highpass.
# The broadcast compressor replaces the light compressor rather than stacking it.
VOICE_FILTERS: dict[str, str] = {
    "none": "",
    "rumble": "highpass=f=80:p=2",
    "proximity": "highpass=f=100:p=2,equalizer=f=200:t=q:w=1:g=-3",
    "proximity_strong": "highpass=f=100:p=2,equalizer=f=200:t=q:w=0.8:g=-5",
    "room": "agate=threshold=0.02:ratio=1.3:range=0.5:attack=10:release=200:makeup=1",
    "harshness": "deesser=i=0.2:m=0.25:f=0.5,highshelf=f=5500:g=-2",
    "clean": (
        "highpass=f=80:p=2,apad=pad_len=1200,afftdn=nr=6:nf=-50,"
        "atrim=start_sample=1200,asetpts=N/SR/TB,"
        "acompressor=threshold=0.125:ratio=1.5:attack=15:release=150:makeup=1"
    ),
    "broadcast": (
        "highpass=f=80:p=2,apad=pad_len=1200,afftdn=nr=6:nf=-50,"
        "atrim=start_sample=1200,asetpts=N/SR/TB,equalizer=f=3000:t=q:w=0.7:g=1.5,"
        "acompressor=threshold=0.1:ratio=2.5:attack=10:release=150:makeup=1"
    ),
    "warm": (
        "highpass=f=80:p=2,equalizer=f=200:t=q:w=0.7:g=1.5,highshelf=f=5500:g=-2,"
        "acompressor=threshold=0.125:ratio=1.5:attack=15:release=150:makeup=1"
    ),
}


def voice_filters(preset: str, *, rumble: bool = False, room: bool = False,
                  harshness: bool = False, proximity: str = "off") -> str:
    """Apply repairs in fixed order before the preset, omitting repeated filters.

    All chains expect 48 kHz audio. An empty treatment returns an empty string.
    """
    if preset not in {"none", "clean", "broadcast", "warm"}:
        raise YapError("Choose voice preset none, clean, broadcast or warm.",
                       error_type="audio_error", code="voice_chain_invalid",
                       suggested_action="Correct the voice preset in the style.")
    if proximity not in ("off", "gentle", "strong"):
        raise YapError("Choose proximity repair off, gentle or strong.",
                       error_type="audio_error", code="voice_chain_invalid",
                       suggested_action="Correct the proximity repair in the style.")
    proximity_name = "proximity_strong" if proximity == "strong" else "proximity"
    names = [name for name, enabled in (("rumble", rumble),
                                        (proximity_name, proximity != "off"), ("room", room),
                                        ("harshness", harshness)) if enabled]
    filters = [part for name in [*names, preset] for part in VOICE_FILTERS[name].split(",")
               if part and not (proximity != "off" and part == "highpass=f=80:p=2")]
    return ",".join(dict.fromkeys(filters))
