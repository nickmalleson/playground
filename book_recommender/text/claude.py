"""Anthropic API integration for generating fresh book picks."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from typing import Optional

from data import Book, HistoryEntry, VALID_GENRES
from state import State, append_log, now_iso


MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
PER_CALL_TIMEOUT = 90.0
BATCH_SIZE = 6


SYSTEM_PROMPT = """You recommend books. The user gives you a list of their original favourite books and authors, a list of titles to avoid, and a list of "slots to replace" — each slot has a genre and a mark indicating how the user reacted to the book it's replacing. For each slot, return one new recommendation.

Output format: your entire reply must be a single JSON array. The first character must be [ and the last must be ]. No prose, no preamble, no code fences, no commentary.

Each element must be an object with exactly these fields:
{"title": string, "author": string, "genre": "cli-fi"|"cyber"|"space"|"fantasy"|"historical"|"literary", "why": string}

The "why" field is 1-2 short sentences tying the pick to a specific original favourite or to the book being replaced.

Rules per replacement:
- "genre" must equal the slot's genre exactly
- The book must NOT appear in originalFavourites or avoidTitles
- mark = "loved": pick something stylistically adjacent (same vibe, voice, themes)
- mark = "passed": pick something in the same genre but with a clearly different style or approach
- mark = "read": pick a strong adjacent book that broadens exposure
- Lesser-known, translated, and out-of-print picks are welcome"""


class APIError(Exception):
    """Wraps anthropic SDK errors with classification info."""
    def __init__(self, message: str, kind: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.kind = kind  # "auth", "rate_limit", "timeout", "network", "parse", "other"
        self.retry_after = retry_after


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "book"


def _unique_id(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # remove leading ```lang? and trailing ```
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _find_balanced_array(text: str) -> Optional[str]:
    """Scan for the first balanced [...] block, respecting JSON string state."""
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start:i + 1]
    return None


def parse_response(raw: str) -> list[dict]:
    """Forgiving parse of Claude's response into a list of pick dicts."""
    cleaned = _strip_code_fences(raw)

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
        raise APIError("Response was a JSON object with no list-valued field.", kind="parse")

    block = _find_balanced_array(cleaned)
    if block is None:
        raise APIError("No JSON array found in response.", kind="parse")
    try:
        result = json.loads(block)
    except json.JSONDecodeError as e:
        raise APIError(f"Could not parse balanced block: {e}", kind="parse")
    if not isinstance(result, list):
        raise APIError("Parsed balanced block is not a list.", kind="parse")
    return result


def build_user_payload(
    state: State,
    original_favourites: list[str],
    slots: list[tuple[Book, str]],
) -> str:
    """Build the user message JSON for one batch."""
    avoid_current = [f"{b.title} by {b.author}" for b in state.current_books]
    avoid_history = [f"{h.title} by {h.author}" for h in state.history]
    loved = [
        f"{b.title} by {b.author}" for b in state.current_books
        if state.marks.get(b.id) and state.marks[b.id].like == "love"
    ] + [
        f"{h.title} by {h.author}" for h in state.history if h.mark == "loved"
    ]
    passed = [
        f"{b.title} by {b.author}" for b in state.current_books
        if state.marks.get(b.id) and state.marks[b.id].like == "meh"
    ] + [
        f"{h.title} by {h.author}" for h in state.history if h.mark == "passed"
    ]
    payload = {
        "originalFavourites": original_favourites,
        "avoidTitles": avoid_current + avoid_history,
        "lovedSoFar": loved,
        "passedSoFar": passed,
        "slotsToReplace": [
            {"genre": b.genre, "replacing": f"{b.title} by {b.author}", "mark": mark_label}
            for (b, mark_label) in slots
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def call_claude(
    client,
    user_payload: str,
    verbose: int = 0,
    model: str = MODEL,
) -> tuple[str, float]:
    """Call the API. Returns (raw_text, latency_seconds). Raises APIError on failure."""
    # Lazy import so --dry-run works without anthropic installed cleanly.
    import anthropic

    start = time.time()
    error_class = None
    error_message = None
    raw = ""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_payload}],
            timeout=PER_CALL_TIMEOUT,
        )
        # response.content is a list of blocks
        if not response.content:
            raise APIError("Empty response (no content blocks).", kind="parse")
        block = response.content[0]
        raw = getattr(block, "text", "") or ""
        latency = time.time() - start
    except anthropic.AuthenticationError as e:
        latency = time.time() - start
        error_class, error_message = type(e).__name__, str(e)
        append_log({
            "model": model, "prompt_chars": len(user_payload), "response_chars": 0,
            "latency_ms": int(latency * 1000), "error": f"{error_class}: {error_message}",
        })
        raise APIError(f"Authentication failed: {e}", kind="auth")
    except anthropic.RateLimitError as e:
        latency = time.time() - start
        error_class, error_message = type(e).__name__, str(e)
        retry_after = None
        try:
            retry_after = float(getattr(e, "response", None).headers.get("retry-after", "")) if getattr(e, "response", None) else None
        except (ValueError, AttributeError, TypeError):
            retry_after = None
        append_log({
            "model": model, "prompt_chars": len(user_payload), "response_chars": 0,
            "latency_ms": int(latency * 1000), "error": f"{error_class}: {error_message}",
        })
        raise APIError(f"Rate limited: {e}", kind="rate_limit", retry_after=retry_after)
    except anthropic.APITimeoutError as e:
        latency = time.time() - start
        error_class, error_message = type(e).__name__, str(e)
        append_log({
            "model": model, "prompt_chars": len(user_payload), "response_chars": 0,
            "latency_ms": int(latency * 1000), "error": f"{error_class}: {error_message}",
        })
        raise APIError(f"Timeout after {PER_CALL_TIMEOUT}s: {e}", kind="timeout")
    except anthropic.APIConnectionError as e:
        latency = time.time() - start
        error_class, error_message = type(e).__name__, str(e)
        append_log({
            "model": model, "prompt_chars": len(user_payload), "response_chars": 0,
            "latency_ms": int(latency * 1000), "error": f"{error_class}: {error_message}",
        })
        raise APIError(f"Network error: {e}", kind="network")
    except anthropic.APIStatusError as e:
        latency = time.time() - start
        error_class, error_message = type(e).__name__, str(e)
        append_log({
            "model": model, "prompt_chars": len(user_payload), "response_chars": 0,
            "latency_ms": int(latency * 1000), "error": f"{error_class}: {error_message}",
        })
        raise APIError(f"API status error: {e}", kind="other")

    log_entry = {
        "model": model,
        "prompt_chars": len(user_payload),
        "response_chars": len(raw),
        "latency_ms": int(latency * 1000),
    }
    if verbose >= 1:
        log_entry["prompt"] = user_payload
        log_entry["response"] = raw
    append_log(log_entry)

    return raw, latency


def validate_pick(pick: dict, slot_genre: str) -> Optional[dict]:
    """Validate and normalize a pick. Returns cleaned pick or None if missing fields."""
    if not isinstance(pick, dict):
        return None
    title = pick.get("title")
    author = pick.get("author")
    why = pick.get("why")
    genre = pick.get("genre")
    if not (isinstance(title, str) and title.strip()):
        return None
    if not (isinstance(author, str) and author.strip()):
        return None
    if not (isinstance(why, str) and why.strip()):
        return None
    # Force genre to match the slot, per spec.
    if genre not in VALID_GENRES or genre != slot_genre:
        genre = slot_genre
    return {
        "title": title.strip(),
        "author": author.strip(),
        "genre": genre,
        "why": why.strip(),
    }


def make_replacement_book(pick: dict, slot_genre: str, taken_ids: set[str]) -> Book:
    base = _slugify(pick["title"])
    bid = _unique_id(base, taken_ids)
    return Book(
        id=bid,
        title=pick["title"],
        author=pick["author"],
        genre=slot_genre,
        why=pick["why"],
        is_fresh=True,
    )


def commit_replacements(
    state: State,
    pairs: list[tuple[Book, dict]],
) -> list[tuple[Book, Book, str]]:
    """For each (old_book, validated_pick), update state in place.
    Returns list of (old_book, new_book, mark_label)."""
    results = []
    taken = {b.id for b in state.current_books}
    for old, pick in pairs:
        mark = state.get_mark(old.id)
        label = mark.label() or "read"
        # History entry for the replaced book
        state.history.insert(0, HistoryEntry(
            title=old.title,
            author=old.author,
            genre=old.genre,
            mark=label,
            replaced_at=now_iso(),
        ))
        # Build replacement
        taken.discard(old.id)
        new_book = make_replacement_book(pick, old.genre, taken)
        taken.add(new_book.id)
        # Replace in current_books at the same position
        for idx, b in enumerate(state.current_books):
            if b.id == old.id:
                state.current_books[idx] = new_book
                break
        # Drop the mark
        state.marks.pop(old.id, None)
        results.append((old, new_book, label))
    return results


def make_placeholder_pick(old: Book) -> dict:
    """For --dry-run: a deterministic fake pick."""
    return {
        "title": f"Placeholder for {old.title}",
        "author": "Dry Run Author",
        "genre": old.genre,
        "why": f"Dry-run replacement for '{old.title}' — no API call was made.",
    }
