"""Make only approved pictures within a fresh spend ceiling."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path

from yap_studio import ffmpeg, paths, receipt
from yap_studio.errors import ValidationError, YapError
from yap_studio.media.catalog import add_generated_item
from yap_studio.pictures import guard, wavespeed
from yap_studio.pictures.attempts import load_attempts, save_attempt
from yap_studio.pictures.metadata import FIELDS as GENERATED_FIELDS
from yap_studio.pictures.sheet import (
    REQUESTS_SCHEMA,
    SCHEMA,
    _lines,
    money,
    require_allowed,
    unavailable,
)
from yap_studio.project.lock import project_lock
from yap_studio.project.model import load_cut_words
from yap_studio.project.status import ProjectContext
from yap_studio.stock.metadata import timestamp
from yap_studio.style import resolve
from yap_studio.taste import apply_taste

MAKE_SCHEMA = "yap-studio/pictures-make@1"
MAX_PICKS = 3
MAX_CEILING_USD = 10.0
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _load_sheet(project_dir: Path, sheet: str | None) -> dict:
    sheets = project_dir / "pictures/sheets"
    if sheet is None:
        numbers = (
            [
                int(p.name[1:])
                for p in sheets.iterdir()
                if p.is_dir() and re.fullmatch(r"s[1-9][0-9]*", p.name)
            ]
            if sheets.exists()
            else []
        )
        if not numbers:
            raise ValidationError("sheet", "no ask sheet; run yap-studio pictures ask first")
        sheet = f"s{max(numbers)}"
    if not isinstance(sheet, str) or re.fullmatch(r"s[1-9][0-9]*", sheet) is None:
        raise ValidationError("sheet", "expected sN")
    try:
        data = json.loads(paths.validate_input(sheets / sheet / "ask-sheet.json").read_bytes())
    except (OSError, ValueError, YapError) as exc:
        raise ValidationError("sheet", f"cannot read {sheet}") from exc
    if (
        not isinstance(data, dict)
        or data.get("schema") != SCHEMA
        or data.get("status") != "ok"
        or data.get("sheet") != sheet
        or data.get("model") != wavespeed.MODEL_ID
        or data.get("size") != wavespeed.SIZE
        or not isinstance(data.get("lines"), list)
    ):
        raise ValidationError("sheet", "invalid ask sheet")
    return data


def _picks(data, numbers):
    by_number = {str(line["number"]): line for line in data["lines"]}
    if not numbers:
        raise ValidationError("pick", "choose at least one number")
    if len(numbers) > MAX_PICKS:
        raise ValidationError(
            "pick", "make at most 3 pictures at a time; show them before making more"
        )
    if any(not isinstance(n, str) or n not in by_number for n in numbers):
        raise ValidationError("pick", "unknown line number")
    if len(numbers) != len(set(numbers)):
        raise ValidationError("pick", "the same number cannot be picked twice")
    return [by_number[n] for n in numbers]


def _ceiling(value) -> Decimal:
    try:
        value = wavespeed.number(value)
        if not 0 < value <= MAX_CEILING_USD:
            raise ValueError("out of range")
    except (ValueError, TypeError):
        raise ValidationError(
            "max_cost", "expected a finite number above 0 and at most 10 USD"
        ) from None
    return Decimal(str(value))


def _cached(project_dir, records, sheet, number):
    for record in records:
        if record.get("sheet") != sheet or str(record.get("number")) != str(number):
            continue
        try:
            target = project_dir / record["path"]
            verified = (
                target.resolve().is_relative_to((project_dir / "pictures/files").resolve())
                and not target.is_symlink()
                and target.is_file()
                and receipt.sha256_file(target) == record["sha256"]
            )
            if not verified:
                continue
            item = add_generated_item(
                target, generated={field: record[field] for field in GENERATED_FIELDS},
                description_text=record["prompt"],
            )
            if item["id"] != record["item"]:
                return {**record, "item": item["id"],
                        "request": {**record["request"], "item": item["id"]}}
            return record
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return None


def _prediction(data):
    prediction = data.get("prediction", {})
    prediction = prediction.get("data", {}) if isinstance(prediction, dict) else {}
    return prediction if isinstance(prediction, dict) else {}


def _download(data, staging):
    prediction = _prediction(data)
    if prediction.get("status") not in {"completed", "succeeded"}:
        raise ValidationError(
            "prediction",
            wavespeed.redact(str(prediction.get("error") or "picture did not complete")),
        )
    if wavespeed.prediction_id(prediction.get("id")) is None:
        raise ValidationError("prediction.id", "invalid prediction id")
    try:
        candidate = Path(data["downloads"][0]["path"])
        source = candidate.resolve()
        if (
            candidate.is_symlink()
            or not source.is_relative_to(staging.resolve())
            or source.suffix.lower() not in IMAGE_SUFFIXES
            or not source.is_file()
            or source.stat().st_size == 0
        ):
            raise ValueError("invalid downloaded file")
    except (KeyError, IndexError, TypeError, ValueError):
        raise ValidationError("download", "expected a non-empty picture inside staging") from None
    info = ffmpeg.probe(source, count_frames=False)
    if (
        not info.has_video
        or info.video_codec not in {"png", "mjpeg", "webp"}
        or not info.width
        or not info.height
        or min(info.width, info.height) <= 0
    ):
        raise ValidationError("download", "expected a picture stream with positive dimensions")
    return source


def _cost(identifier, price, estimate):
    charged = wavespeed.charged(identifier) if identifier else None
    cost = charged if charged is not None else price if price is not None else estimate
    return charged, cost, "charged" if charged is not None else "estimate"


def _record(project_dir, data, line, source, identifier, estimate, charged, cost, cost_source, now):
    target = project_dir / "pictures/files" / (identifier + source.suffix.lower())
    paths.validate_output_dir(target.parent)
    paths._guard_write(target)
    os.replace(source, target)
    at = timestamp(now)
    generated = {
        "provider": wavespeed.PROVIDER,
        "model": wavespeed.MODEL_ID,
        "prompt": line["prompt"],
        "seed": None,
        "size": wavespeed.SIZE,
        "prediction_id": identifier,
        "cost_usd": cost,
        "cost_source": cost_source,
        "created_at": at,
    }
    item = add_generated_item(target, generated=generated, description_text=line["prompt"], now=now)
    request = {
        "id": "broll-" + line["id"],
        "item": item["id"],
        "kind": "insert",
        "start": line["start"],
        "end": line["end"],
        "audio": "mute",
    }
    record = {
        "sheet": data["sheet"],
        "number": line["number"],
        "line": line["id"],
        "text": line["text"],
        "prompt": line["prompt"],
        "item": item["id"],
        "sha256": item["sha256"],
        "path": str(target.relative_to(project_dir)),
        **generated,
        "estimate_usd": estimate,
        "charged_usd": charged,
        "approved_at": at,
        "request": request,
    }
    with project_lock(project_dir):
        records = [
            p
            for p in guard.approved_pictures(project_dir)
            if (p.get("sheet"), p.get("number")) != (data["sheet"], line["number"])
        ]
        records.append(record)
        paths.atomic_write_text(
            project_dir / "pictures/pictures.json",
            json.dumps(
                {"schema": guard.PICTURES_SCHEMA, "pictures": records}, indent=2, allow_nan=False
            )
            + "\n",
        )
    return record


def _result(data, ceiling):
    return {
        "schema": MAKE_SCHEMA,
        "status": "ok",
        "sheet": data["sheet"],
        "made": [],
        "recovered": [],
        "cached": [],
        "failed": [],
        "requests": [],
        "spent_usd": 0.0,
        "ceiling_usd": float(ceiling),
    }


def _remaining(project_dir, data, chosen, result):
    records = guard.approved_pictures(project_dir)
    remaining = []
    for line in chosen:
        cached = _cached(project_dir, records, data["sheet"], line["number"])
        if cached:
            result["cached"].append(cached)
            result["requests"].append(cached["request"])
        else:
            remaining.append(line)
    return remaining


def _stale(line, sheet):
    return ValidationError(
        "pick", f"line {line['number']} changed since sheet {sheet}; run pictures ask again"
    )


def _validate_lines(project_dir, data, chosen):
    try:
        source = Path(data.get("cut_words", "cut/cut-words.json"))
        if not source.is_absolute():
            source = project_dir / source
        words = load_cut_words(paths.validate_input(source))
    except (YapError, OSError, ValueError, TypeError):
        raise _stale(chosen[0], data["sheet"]) from None
    for line in chosen:
        try:
            request = {key: line[key] for key in ("id", "start", "end", "prompt")}
            current = _lines({"schema": REQUESTS_SCHEMA, "lines": [request]}, words)[0]
            if current["text"] != line["text"] or (
                "word_ids" in line and current["word_ids"] != line["word_ids"]
            ):
                raise ValueError("line changed")
        except (YapError, ValueError, KeyError, TypeError):
            raise _stale(line, data["sheet"]) from None


def _failure(line, reason, identifier=None):
    failure = {"number": line["number"], "reason": reason}
    if identifier:
        failure.update(
            prediction_id=identifier, recover=f"wavespeed-pp-cli prediction-results {identifier}"
        )
    return failure


def _recover_payload(project_dir, data, line, attempt, staging):
    identifier = attempt["prediction_id"]
    payload = {"prediction": wavespeed.prediction_results(identifier)}
    prediction = _prediction(payload)
    if wavespeed.prediction_failed(prediction):
        save_attempt(project_dir, data["sheet"], line["number"], None)
        raise wavespeed.GenerationError(
            "WaveSpeed reported the earlier attempt failed; "
            "run pictures make again to try once more.",
            identifier=identifier,
            failed=True,
        )
    if prediction.get("status") not in {"completed", "succeeded"}:
        raise wavespeed.GenerationError(
            "WaveSpeed prediction is still processing.", identifier=identifier
        )
    outputs = prediction.get("outputs")
    if not isinstance(outputs, list) or not outputs or not isinstance(outputs[0], str):
        raise wavespeed.GenerationError(
            "WaveSpeed returned no picture output.", identifier=identifier
        )
    payload["downloads"] = wavespeed.download(outputs[0], staging).get("downloads")
    return payload


def _obtain(project_dir, data, line, quote, attempt, now):
    staging = Path(tempfile.mkdtemp(prefix=".make-", dir=project_dir / "pictures"))
    recovering = attempt is not None
    identifier = attempt["prediction_id"] if recovering else None
    evidence = attempt or {
        "sheet": data["sheet"],
        "number": line["number"],
        "prediction_id": None,
        "quote_usd": quote,
        "started_at": timestamp(now),
    }
    spent, charged, accounted = 0.0, None, False
    uncertain = False
    try:
        if recovering:
            payload = _recover_payload(project_dir, data, line, evidence, staging)
        else:
            save_attempt(project_dir, data["sheet"], line["number"], evidence)
            uncertain = True
            payload = wavespeed.generate(line["prompt"], staging, dry_run=False)
            identifier = wavespeed.prediction_id(_prediction(payload).get("id"))
            uncertain = identifier is None
            if identifier:
                evidence["prediction_id"] = identifier
                save_attempt(project_dir, data["sheet"], line["number"], evidence)
        price = None if recovering else wavespeed.run_price(payload)
        charged, cost, cost_source = _cost(identifier, price, quote)
        spent = cost if identifier and not recovering else 0.0
        accounted = True
        if wavespeed.prediction_failed(_prediction(payload)):
            uncertain = False
            save_attempt(project_dir, data["sheet"], line["number"], None)
            raise ValidationError(
                "prediction",
                wavespeed.safe_detail(
                    str(_prediction(payload).get("error") or "WaveSpeed prediction failed."),
                    prompt=line["prompt"],
                    staging_dir=staging,
                ),
            )
        source = _download(payload, staging)
        record = _record(
            project_dir, data, line, source, identifier, quote, charged, cost, cost_source, now
        )
        save_attempt(project_dir, data["sheet"], line["number"], None)
        return {"entry": record, "failure": None, "spent": spent, "charged": charged,
                "uncertain": False}
    except (YapError, OSError, ValueError, TypeError) as exc:
        if isinstance(exc, wavespeed.GenerationError) and not recovering:
            identifier = exc.prediction_id
            if identifier:
                uncertain = False
                evidence["prediction_id"] = identifier
                try:
                    save_attempt(project_dir, data["sheet"], line["number"], evidence)
                except (YapError, OSError):
                    pass
            if exc.failed:
                uncertain = False
                save_attempt(project_dir, data["sheet"], line["number"], None)
        if identifier and not recovering and not accounted:
            charged, spent, _ = _cost(identifier, getattr(exc, "price", None), quote)
        reason = wavespeed.safe_detail(str(exc), prompt=line["prompt"], staging_dir=staging)
        return {
            "entry": None,
            "failure": _failure(line, reason or "Picture could not be saved.", identifier),
            "spent": quote if uncertain else spent,
            "charged": charged,
            "uncertain": uncertain,
        }
    finally:
        shutil.rmtree(staging)


def _make(project_dir, data, chosen, ceiling, now, new):
    result = _result(data, ceiling)
    remaining = _remaining(project_dir, data, chosen, result)
    if not remaining:
        return result
    _validate_lines(project_dir, data, remaining)
    prior = {(entry["sheet"], entry["number"]): entry for entry in load_attempts(project_dir)}
    pending = []
    for line in remaining:
        attempt = None if new else prior.get((data["sheet"], line["number"]))
        if attempt is not None and attempt["prediction_id"] is None:
            result["failed"].append(
                _failure(
                    line,
                    (
                        f"A previous attempt for line {line['number']} may have been charged. "
                        "Check wavespeed-pp-cli predictions, then pass --new to make it again."
                    ),
                )
            )
        else:
            pending.append((line, attempt))
    if not pending:
        return result
    state = wavespeed.status()
    if state != "ready":
        return unavailable(state, schema=MAKE_SCHEMA)
    try:
        quotes = [
            attempt["quote_usd"] if attempt else wavespeed.estimate(line["prompt"])
            for line, attempt in pending
        ]
    except wavespeed.PicturesUnavailable as exc:
        return unavailable(exc.reason, schema=MAKE_SCHEMA)
    quoted = sum(
        (
            Decimal(str(quote))
            for quote, (_, attempt) in zip(quotes, pending, strict=True)
            if attempt is None
        ),
        Decimal(0),
    )
    if quoted > ceiling:
        raise YapError(
            f"These pictures would cost about {money(quoted)}, over your {money(ceiling)} ceiling.",
            code="pictures_over_ceiling",
        )
    spent = Decimal(0)
    stopped = None
    for (line, attempt), quote in zip(pending, quotes, strict=True):
        if stopped is None and guard.imagery_answer(project_dir) != "yes":
            stopped = "stopped: the creator no longer wants generated pictures for this video"
        if stopped is None and attempt is None and spent + Decimal(str(quote)) > ceiling:
            stopped = f"stopped at the {money(ceiling)} spend ceiling"
        if stopped:
            result["failed"].append(_failure(line, stopped))
            continue
        obtained = _obtain(project_dir, data, line, quote, attempt, now)
        spent += Decimal(str(obtained["spent"]))
        if obtained["failure"]:
            result["failed"].append(obtained["failure"])
        else:
            entry = obtained["entry"]
            result["recovered" if attempt else "made"].append(entry)
            result["requests"].append(entry["request"])
        charged = obtained["charged"]
        if obtained["uncertain"]:
            stopped = (
                f"stopped: line {line['number']} may have been charged. "
                "Check wavespeed-pp-cli predictions, then pass --new to make it again."
            )
        elif (
            attempt is None and charged is not None
            and Decimal(str(charged)) > Decimal(str(quote))
        ):
            stopped = (
                f"stopped: WaveSpeed charged {money(charged)} for line {line['number']}, "
                f"more than its {money(quote)} quote"
            )
    result["spent_usd"] = float(spent)
    return result


def make_pictures(
    project_dir, numbers, *, yes, max_cost, sheet=None, dry_run=False, now=None, new=False
) -> dict:
    """Validate consent and every free preflight before spending."""
    if not yes and not dry_run:
        raise YapError(
            "Making pictures needs the creator's yes.",
            code="pictures_need_yes",
            suggested_action="Show the ask sheet with its costs, ask, then pass "
            "--yes and --max-cost.",
        )
    project_dir = Path(project_dir).resolve()
    with paths.operation():
        style = (
            ProjectContext(project_dir).render_style().data
            if (project_dir / "project.json").is_file()
            else apply_taste(resolve(None)).style.data
        )
        require_allowed(style, project_dir)
        data = _load_sheet(project_dir, sheet)
        chosen = _picks(data, numbers)
        ceiling = None if dry_run else _ceiling(max_cost)
        if dry_run:
            state = wavespeed.status()
            if state != "ready":
                return unavailable(state, schema=MAKE_SCHEMA)
            return {
                "schema": MAKE_SCHEMA,
                "status": "dry_run",
                "sheet": data["sheet"],
                "previews": [
                    {
                        "number": line["number"],
                        "prompt": line["prompt"],
                        "request": wavespeed.generate(
                            line["prompt"], project_dir / "pictures/preview", dry_run=True
                        ),
                    }
                    for line in chosen
                ],
            }
        result = _result(data, ceiling)
        if not _remaining(project_dir, data, chosen, result):
            return result
        # Serialize paid calls and recovery, then recheck the catalog inside the lock.
        with project_lock(project_dir / "pictures"):
            return _make(project_dir, data, chosen, ceiling, now, new)
