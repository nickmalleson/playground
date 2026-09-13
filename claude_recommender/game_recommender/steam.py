"""Steam review ratings for game titles.

Two public, no-key Steam store endpoints:
    storesearch   title -> candidate apps (Steam's own ranking)
    appreviews    appid -> review totals and the "Very Positive"-style label

Titles come from Claude as free text, so matching is deliberately strict:
an exact match after normalisation, or the same title plus an edition suffix
("Definitive Edition", ": PC Edition", ...). Sequels, DLC and soundtracks are
rejected rather than risk showing the wrong game's rating.

Results are cached on disk keyed by lowercased title. Misses are cached too
(with a TTL) so a game that isn't on Steam doesn't get re-queried every load.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("recommender.steam")

SEARCH_URL = "https://store.steampowered.com/api/storesearch/?cc=gb&l=en&term="
REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}?json=1&language=all&purchase_type=all&num_per_page=0"
STORE_URL = "https://store.steampowered.com/app/{appid}"
TIMEOUT_S = 5.0
MISS_TTL_S = 7 * 24 * 3600

# Words that may trail a title and still mean "the same game"
_EDITION_WORDS = {
    "pc", "edition", "definitive", "remastered", "remaster", "enhanced", "deluxe",
    "complete", "ultimate", "anniversary", "goty", "game", "of", "the", "year",
    "directors", "director's", "cut", "special", "hd", "collection", "gold",
}


def _normalise(s: str) -> str:
    s = s.lower().replace("™", "").replace("®", "").replace("©", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def pick_match(title: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the store-search item that is the same game as `title`, or None."""
    want = _normalise(title)
    if not want:
        return None

    edition_hit = None
    for item in items:
        have = _normalise(str(item.get("name", "")))
        if have == want:
            return item
        if edition_hit is None and have.startswith(want + " "):
            rest = have[len(want):].split()
            if rest and all(w in _EDITION_WORDS for w in rest):
                edition_hit = item
    if edition_hit is not None:
        return edition_hit

    # Typo tolerance only: very high bar so sequels/DLC never slip through
    for item in items:
        have = _normalise(str(item.get("name", "")))
        if difflib.SequenceMatcher(None, want, have).ratio() >= 0.95:
            return item
    return None


def _fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "game-recommender/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.load(resp)


def lookup(title: str) -> dict[str, Any] | None:
    """Look a title up on Steam. Returns a rating record, or None if unmatched,
    unreviewed, or the network failed (errors are logged, never raised)."""
    try:
        search = _fetch_json(SEARCH_URL + urllib.parse.quote(title))
        item = pick_match(title, search.get("items") or [])
        if not item:
            log.info(f"steam: no confident match for {title!r}")
            return None
        appid = int(item["id"])
        summary = (_fetch_json(REVIEWS_URL.format(appid=appid)) or {}).get("query_summary") or {}
        total = int(summary.get("total_reviews") or 0)
        if total <= 0:
            log.info(f"steam: {title!r} matched appid {appid} but has no reviews")
            return None
        positive = int(summary.get("total_positive") or 0)
        rec = {
            "appid": appid,
            "name": item.get("name"),
            "positive": positive,
            "total": total,
            "pct": round(100 * positive / total),
            "score_desc": summary.get("review_score_desc"),
            "url": STORE_URL.format(appid=appid),
        }
        log.info(f"steam: {title!r} -> {rec['name']!r} {rec['pct']}% of {total}")
        return rec
    except Exception as e:  # network, JSON shape, whatever — never break the app
        log.warning(f"steam: lookup failed for {title!r}: {type(e).__name__}: {e}")
        return None


class SteamCache:
    """Disk-backed title -> rating cache shared across users."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning(f"steam: could not read cache {self.path}: {e}; starting empty")

    def get(self, title: str) -> dict[str, Any] | None:
        key = title.strip().lower()
        with self._lock:
            entry = self._data.get(key)
        if entry is not None:
            if entry.get("rating") is not None:
                return entry["rating"]
            if time.time() - float(entry.get("checked_at", 0)) < MISS_TTL_S:
                return None
        rating = lookup(title)
        with self._lock:
            self._data[key] = {"rating": rating, "checked_at": time.time()}
            self._save()
        return rating

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
