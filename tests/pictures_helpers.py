"""Synthetic per-video imagery answers."""

from test_timeline import write

PROMPT = "Do you want me to create imagery for this video?"


def record_answer(root, answer="yes"):
    return write(root / "pictures/imagery.json", {
        "schema": "yap-studio/imagery-answer@1",
        "question": PROMPT,
        "answer": answer,
        "answered_at": "2026-10-19T12:00:00Z",
    })
