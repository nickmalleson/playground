"""State persistence: load, save, and migrate the JSON state file."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from data import Book, Mark, HistoryEntry, SEED_BOOKS, VALID_GENRES


STATE_DIR = Path.home() / ".book_recommender"
STATE_PATH = STATE_DIR / "state.json"
LOG_PATH = STATE_DIR / "log.jsonl"
STATE_VERSION = 1


class State:
    def __init__(
        self,
        current_books: list[Book],
        marks: dict[str, Mark],
        history: list[HistoryEntry],
    ):
        self.current_books = current_books
        self.marks = marks
        self.history = history

    @classmethod
    def fresh(cls) -> "State":
        return cls(
            current_books=[Book(**asdict(b)) for b in SEED_BOOKS],
            marks={},
            history=[],
        )

    def to_dict(self) -> dict:
        return {
            "version": STATE_VERSION,
            "current_books": [asdict(b) for b in self.current_books],
            "marks": {bid: asdict(m) for bid, m in self.marks.items()},
            "history": [asdict(h) for h in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        books = [Book(**b) for b in d.get("current_books", [])]
        marks_raw = d.get("marks", {})
        marks: dict[str, Mark] = {}
        for bid, m in marks_raw.items():
            marks[bid] = Mark(read=bool(m.get("read", False)), like=m.get("like"))
        history = [HistoryEntry(**h) for h in d.get("history", [])]
        return cls(books, marks, history)

    def get_mark(self, book_id: str) -> Mark:
        return self.marks.get(book_id) or Mark()

    def set_mark(self, book_id: str, mark: Mark) -> None:
        if not mark.is_marked():
            self.marks.pop(book_id, None)
        else:
            self.marks[book_id] = mark

    def marked_books(self) -> list[Book]:
        return [b for b in self.current_books if self.marks.get(b.id) and self.marks[b.id].is_marked()]

    def loved_books(self) -> list[Book]:
        return [b for b in self.current_books if self.marks.get(b.id) and self.marks[b.id].like == "love"]

    def passed_books(self) -> list[Book]:
        return [b for b in self.current_books if self.marks.get(b.id) and self.marks[b.id].like == "meh"]


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> State:
    ensure_state_dir()
    if not STATE_PATH.exists():
        state = State.fresh()
        save_state(state)
        return state
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return State.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = STATE_DIR / f"state.json.corrupt-{ts}"
        try:
            STATE_PATH.rename(backup)
            print(f"[warn] state file corrupt ({e}); backed up to {backup}, starting fresh.", file=sys.stderr)
        except OSError as oe:
            print(f"[warn] state file corrupt and could not back up ({oe}); starting fresh.", file=sys.stderr)
        state = State.fresh()
        save_state(state)
        return state


def save_state(state: State) -> None:
    """Atomic write: write to temp, fsync, rename."""
    ensure_state_dir()
    tmp = STATE_PATH.with_suffix(".json.tmp")
    payload = json.dumps(state.to_dict(), indent=2, ensure_ascii=False)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(STATE_PATH)


def append_log(entry: dict) -> None:
    ensure_state_dir()
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def humanize_age(iso_ts: str) -> str:
    """'replaced 3 days ago' style relative time."""
    try:
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return iso_ts
    delta = datetime.now(timezone.utc) - ts
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        m = secs // 60
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if secs < 86400:
        h = secs // 3600
        return f"{h} hour{'s' if h != 1 else ''} ago"
    days = secs // 86400
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"
    months = days // 30
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"
