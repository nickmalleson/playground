"""
Local web server for the book recommender.

Run with:
    python server.py [--debug] [--port 5050]

Then open http://localhost:5050 in your browser.

Requires:
    - ANTHROPIC_API_KEY environment variable
    - pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Third-party (install via requirements.txt)
try:
    from flask import Flask, jsonify, request, send_from_directory
except ImportError:
    print("ERROR: flask not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

try:
    from anthropic import Anthropic, APIError, APITimeoutError
except ImportError:
    print("ERROR: anthropic not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


# ───────────────────── Configuration ─────────────────────

HERE = Path(__file__).parent.resolve()
STATE_FILE = HERE / "state.json"
LOG_FILE = HERE / "server.log"
DEFAULT_PORT = 5050

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
BATCH_SIZE = 6        # books per Claude call
TIMEOUT_S = 90.0      # per-call timeout


# ───────────────────── Logging ─────────────────────

class ColourFormatter(logging.Formatter):
    """Add a touch of colour to terminal output."""
    GREY = "\033[90m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    LEVEL_COLOUR = {
        "DEBUG": GREY,
        "INFO": BLUE,
        "WARNING": YELLOW,
        "ERROR": RED,
        "CRITICAL": BOLD + RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        colour = self.LEVEL_COLOUR.get(record.levelname, "")
        level = f"{colour}{record.levelname:<5}{self.RESET}"
        return f"{self.GREY}{ts}{self.RESET}  {level}  {record.getMessage()}"


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    # Stream handler (terminal, with colour)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(ColourFormatter())
    root.addHandler(sh)

    # File handler (plain text)
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-5s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)
    except Exception as e:
        print(f"WARNING: could not open log file {LOG_FILE}: {e}", file=sys.stderr)

    # Quiet down werkzeug's per-request access log; we do our own
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


log = logging.getLogger("recommender")


# ───────────────────── Seed data ─────────────────────

ORIGINAL_FAVOURITES = [
    "Margaret Atwood — Oryx and Crake",
    "William Gibson — The Sprawl Trilogy and most other Gibson",
    "Cormac McCarthy — The Road",
    "Tim Winton — Juice",
    "Paolo Bacigalupi — all books, especially The Water Knife and The Windup Girl",
    "George R. R. Martin — A Game of Thrones",
    "Patrick Rothfuss — The Kingkiller Chronicle",
    "Joe Abercrombie — all books, especially The First Law",
    "Cixin Liu — The Three-Body Problem and the rest of the trilogy",
    "Dan Simmons — the Hyperion Cantos",
    "Robert Harris — all books",
    "Bernard Cornwell — especially the Saxon Stories (Uhtred and Alfred the Great)",
    "Ursula K. Le Guin — sci-fi and fantasy",
    "David Mitchell — Cloud Atlas",
    "Frank Herbert — Dune (and posthumous continuations)",
    "Scott Lynch — Gentleman Bastard / Locke Lamora",
]

GENRES = [
    {"key": "cli-fi", "label": "Climate dystopia", "sub": "For when you want Bacigalupi, Atwood or McCarthy"},
    {"key": "cyber", "label": "Cyberpunk and tech-noir", "sub": "For when you want Gibson"},
    {"key": "space", "label": "Space opera and hard sci-fi", "sub": "For when you want Hyperion, Three Body or Dune"},
    {"key": "fantasy", "label": "Epic fantasy", "sub": "For when you want Abercrombie, GRRM or Lynch"},
    {"key": "historical", "label": "Historical fiction", "sub": "For when you want Cornwell or Harris"},
    {"key": "literary", "label": "Literary speculative", "sub": "For when you want Le Guin or Mitchell"},
]

SEED_BOOKS: list[dict[str, Any]] = [
    # Climate dystopia
    {"id": "parable-of-the-sower", "title": "Parable of the Sower", "author": "Octavia E. Butler", "genre": "cli-fi",
     "why": "Walled communities, climate collapse, a teenage prophet on the road. The bone-deep dread of The Road meets the political thrust of Bacigalupi — and it was written in 1993."},
    {"id": "ministry-future", "title": "The Ministry for the Future", "author": "Kim Stanley Robinson", "genre": "cli-fi",
     "why": "A polyphonic climate novel that opens with a heatwave scene as devastating as anything in The Water Knife, then pivots into oddly hopeful policy wonkery."},
    {"id": "american-war", "title": "American War", "author": "Omar El Akkad", "genre": "cli-fi",
     "why": "A second US civil war in the 2070s, fought over the last fossil fuels. Spare, devastating prose — distinctly McCarthy-coded, with a slow-burn protagonist arc."},
    {"id": "gold-fame-citrus", "title": "Gold Fame Citrus", "author": "Claire Vaye Watkins", "genre": "cli-fi",
     "why": "California's water is gone and a dune sea is swallowing the Southwest. Lyrical and hallucinatory — basically a literary sibling to The Water Knife."},

    # Cyberpunk
    {"id": "snow-crash", "title": "Snow Crash", "author": "Neal Stephenson", "genre": "cyber",
     "why": "Cyberpunk's other foundational text — funnier and faster than Gibson. Samurai pizza couriers, a metaverse linguistic virus, and the best opening 50 pages in the genre."},
    {"id": "altered-carbon", "title": "Altered Carbon", "author": "Richard Morgan", "genre": "cyber",
     "why": "Hard-boiled detective noir plus downloadable consciousness. The Sprawl trilogy's 2000s grandchild — grittier, pulpier, and very confident."},
    {"id": "quantum-thief", "title": "The Quantum Thief", "author": "Hannu Rajaniemi", "genre": "cyber",
     "why": "A post-cyberpunk caper across a transhuman solar system. Dense, dazzling, throws you in the deep end exactly the way Neuromancer did."},
    {"id": "blindsight", "title": "Blindsight", "author": "Peter Watts", "genre": "cyber",
     "why": "First-contact horror with transhumans and predator-vampires interrogating consciousness itself. Brutally smart — Hyperion fans tend to love it too."},

    # Space opera & hard SF
    {"id": "memory-called-empire", "title": "A Memory Called Empire", "author": "Arkady Martine", "genre": "space",
     "why": "A small-station ambassador navigates the seductive horror of a vast empire. Reads like Hyperion crossed with Le Guin's anthropological eye. The sequel is just as good."},
    {"id": "player-of-games", "title": "The Player of Games", "author": "Iain M. Banks", "genre": "space",
     "why": "The friendliest doorway into the Culture. Scope, civilisational politics, and moral weight that should hit your Dune and Hyperion nerve. Banks is the missing author from your list."},
    {"id": "children-of-time", "title": "Children of Time", "author": "Adrian Tchaikovsky", "genre": "space",
     "why": "Uplifted spiders evolve over millennia while a battered generation ship limps toward them. Big-idea SF in Cixin Liu's lineage but with more empathy."},
    {"id": "house-of-suns", "title": "House of Suns", "author": "Alastair Reynolds", "genre": "space",
     "why": "A six-million-year galactic timescale, with clones of a single woman trading memories at meet-ups every 200,000 years. Hyperion-grade scope and melancholy."},

    # Epic fantasy
    {"id": "gardens-of-the-moon", "title": "Gardens of the Moon (Malazan)", "author": "Steven Erikson", "genre": "fantasy",
     "why": "An army-scale, mythologically dense fantasy spread across continents and millennia. Steeper learning curve than First Law but the payoff over ten books is enormous."},
    {"id": "black-company", "title": "The Black Company", "author": "Glen Cook", "genre": "fantasy",
     "why": "Grimdark before grimdark was a word. Mercenary-company POV, blood-and-mud morality — the obvious ancestor of Abercrombie's First Law."},
    {"id": "assassins-apprentice", "title": "Assassin's Apprentice", "author": "Robin Hobb", "genre": "fantasy",
     "why": "Slow-burn, ferocious character work across a 16-book braided saga. The author Rothfuss readers most often graduate to — and she actually finishes her series."},
    {"id": "poppy-war", "title": "The Poppy War", "author": "R. F. Kuang", "genre": "fantasy",
     "why": "Military academy to genocide-scale war, drawn from 20th-century Chinese history. Brutal, modern, GRRM-territory stakes — not for the faint-hearted."},

    # Historical fiction
    {"id": "wolf-hall", "title": "Wolf Hall", "author": "Hilary Mantel", "genre": "historical",
     "why": "Thomas Cromwell's rise in Henry VIII's court. Prose good enough to be illegal, plus all the political cunning of GRRM but in actual history. The trilogy is one of the great achievements."},
    {"id": "shogun", "title": "Shōgun", "author": "James Clavell", "genre": "historical",
     "why": "An Englishman shipwrecked into Sengoku-era Japan. Massive, addictive, written with the same propulsive instinct as Cornwell's Saxon books."},
    {"id": "master-and-commander", "title": "Master and Commander", "author": "Patrick O'Brian", "genre": "historical",
     "why": "Napoleonic naval fiction across 20 books. The more literary cousin of Cornwell's Sharpe — quieter, funnier, profoundly companionable once you tune in to the rhythm."},
    {"id": "hhhh", "title": "HHhH", "author": "Laurent Binet", "genre": "historical",
     "why": "The 1942 plot to assassinate Reinhard Heydrich, told as a meta-narrative about the difficulty of telling it. Slim, riveting — Robert Harris would adore this one."},

    # Literary speculative
    {"id": "piranesi", "title": "Piranesi", "author": "Susanna Clarke", "genre": "literary",
     "why": "A man lives alone in an infinite labyrinth of statues and tides. Short, strange, perfect — Le Guin would have loved it."},
    {"id": "never-let-me-go", "title": "Never Let Me Go", "author": "Kazuo Ishiguro", "genre": "literary",
     "why": "Quiet dystopia disguised as a boarding-school memoir. Devastating in the same way Cloud Atlas is devastating, but more concentrated."},
    {"id": "jonathan-strange", "title": "Jonathan Strange & Mr Norrell", "author": "Susanna Clarke", "genre": "literary",
     "why": "Two magicians revive English magic during the Napoleonic Wars. Footnoted, funny, immense — the texture of Le Guin with the wit of an Austen novel."},
    {"id": "memory-police", "title": "The Memory Police", "author": "Yoko Ogawa", "genre": "literary",
     "why": "On a small island, things vanish from memory one by one — birds, ribbons, novels, body parts. Ishiguro-quiet, Atwood-political, hauntingly beautiful."},
]


# ───────────────────── State management ─────────────────────

def default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "current_books": [dict(b) for b in SEED_BOOKS],
        "marks": {},
        "history": [],
    }


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        log.info(f"No state file at {STATE_FILE} — creating fresh state")
        st = default_state()
        save_state(st)
        return st

    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            st = json.load(f)
        # Validate shape
        for key in ("current_books", "marks", "history"):
            if key not in st:
                raise ValueError(f"missing key: {key}")
        return st
    except Exception as e:
        backup = STATE_FILE.with_suffix(f".corrupt-{int(time.time())}.json")
        log.error(f"State file corrupt ({e}). Backing up to {backup.name} and starting fresh.")
        try:
            STATE_FILE.rename(backup)
        except Exception as ee:
            log.error(f"Could not rename corrupt state file: {ee}")
        st = default_state()
        save_state(st)
        return st


def save_state(state: dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(STATE_FILE)
    log.debug(
        f"saved state: {len(state['current_books'])} books, "
        f"{len(state['marks'])} marks, {len(state['history'])} history"
    )


# ───────────────────── Claude integration ─────────────────────

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
- Lesser-known, translated, and out-of-print picks are welcome
"""


def call_claude(payload: dict[str, Any]) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    client = Anthropic(api_key=api_key, timeout=TIMEOUT_S)
    user_content = json.dumps(payload, indent=2, ensure_ascii=False)

    log.info(
        f"→ Claude  model={MODEL}  prompt_chars={len(user_content)}  "
        f"slots={len(payload.get('slotsToReplace', []))}"
    )
    log.debug(f"full prompt:\n{user_content}")

    t0 = time.time()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    except APITimeoutError as e:
        elapsed = time.time() - t0
        log.error(f"× Claude timed out after {elapsed:.1f}s: {e}")
        raise
    except APIError as e:
        elapsed = time.time() - t0
        log.error(f"× Claude API error after {elapsed:.1f}s: {type(e).__name__}: {e}")
        raise

    elapsed = time.time() - t0
    text = response.content[0].text
    usage = getattr(response, "usage", None)
    usage_str = ""
    if usage is not None:
        usage_str = f"  in_tok={getattr(usage, 'input_tokens', '?')}  out_tok={getattr(usage, 'output_tokens', '?')}"
    log.info(f"← Claude  latency={elapsed:.1f}s  resp_chars={len(text)}{usage_str}")
    log.debug(f"raw response:\n{text}")

    return text


def parse_picks(text: str) -> list[dict[str, Any]] | None:
    """Forgiving parser: handles fences, object-wrapping, leading commentary."""
    s = text.strip()
    s = re.sub(r"^```(?:json|javascript|js)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s, flags=re.IGNORECASE).strip()

    # Direct parse
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError as e:
        log.debug(f"direct parse failed: {e}")

    # Balanced array extraction
    start = s.find("[")
    if start < 0:
        return None
    depth, in_string, escape = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError as e:
                    log.debug(f"balanced parse failed: {e}")
                    return None
    return None


def make_unique_id(title: str, state: dict[str, Any]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (title or "untitled").lower()).strip("-") or "untitled"
    taken = {b["id"] for b in state["current_books"]} | {h.get("id", "") for h in state["history"]}
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def label_for_mark(m: dict[str, Any]) -> str:
    if m.get("like") == "love":
        return "loved"
    if m.get("like") == "meh":
        return "passed"
    return "read"


# ───────────────────── Flask app ─────────────────────

app = Flask(__name__, static_folder=None)


@app.before_request
def _log_request() -> None:
    log.debug(f"→ {request.method} {request.path}")


@app.after_request
def _log_response(resp):
    log.debug(f"← {resp.status_code} {request.method} {request.path}")
    return resp


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/api/state", methods=["GET"])
def api_get_state():
    state = load_state()
    return jsonify(
        {
            "current_books": state["current_books"],
            "marks": state["marks"],
            "history": state["history"],
            "genres": GENRES,
            "original_favourites": ORIGINAL_FAVOURITES,
            "model": MODEL,
        }
    )


@app.route("/api/mark", methods=["POST"])
def api_mark():
    body = request.get_json(silent=True) or {}
    book_id = body.get("id")
    patch = body.get("patch", {}) or {}
    if not book_id or not isinstance(patch, dict):
        return jsonify({"ok": False, "error": "Missing id or patch"}), 400

    log.info(f"mark  id={book_id!r}  patch={patch}")

    state = load_state()
    cur = state["marks"].get(book_id, {"read": False, "like": None})
    cur = {**cur, **patch}
    if patch.get("like") in ("love", "meh"):
        cur["read"] = True
    state["marks"][book_id] = cur
    save_state(state)
    return jsonify({"ok": True, "marks": state["marks"]})


@app.route("/api/clear-marks", methods=["POST"])
def api_clear_marks():
    log.info("clear all marks")
    state = load_state()
    state["marks"] = {}
    save_state(state)
    return jsonify({"ok": True, "marks": state["marks"]})


@app.route("/api/clear-history", methods=["POST"])
def api_clear_history():
    log.info("clear history")
    state = load_state()
    state["history"] = []
    save_state(state)
    return jsonify({"ok": True, "history": state["history"]})


@app.route("/api/fresh-picks", methods=["POST"])
def api_fresh_picks():
    state = load_state()
    marked_books = [
        b for b in state["current_books"]
        if (m := state["marks"].get(b["id"])) and (m.get("read") or m.get("like"))
    ]
    if not marked_books:
        log.warning("fresh-picks called with no marked books")
        return jsonify({"ok": False, "error": "No marked books to replace."}), 400

    log.info(f"fresh-picks  marked={len(marked_books)}")

    avoid_titles = (
        [f"{b['title']} by {b['author']}" for b in state["current_books"]]
        + [f"{h['title']} by {h['author']}" for h in state["history"]]
    )
    loved_so_far = [
        f"{b['title']} by {b['author']}"
        for b in state["current_books"]
        if state["marks"].get(b["id"], {}).get("like") == "love"
    ]
    passed_so_far = [
        f"{b['title']} by {b['author']}"
        for b in state["current_books"]
        if state["marks"].get(b["id"], {}).get("like") == "meh"
    ]

    slots = []
    for b in marked_books:
        m = state["marks"][b["id"]]
        slots.append(
            {
                "genre": b["genre"],
                "replacing": f"{b['title']} by {b['author']}",
                "mark": label_for_mark(m),
            }
        )

    batches = [slots[i : i + BATCH_SIZE] for i in range(0, len(slots), BATCH_SIZE)]
    log.info(f"split into {len(batches)} batch(es) of up to {BATCH_SIZE}")

    all_picks: list[dict[str, Any]] = []
    last_response = ""
    error: str | None = None

    for bi, batch in enumerate(batches, start=1):
        log.info(f"  batch {bi}/{len(batches)}: {len(batch)} slot(s)")
        payload = {
            "originalFavourites": ORIGINAL_FAVOURITES,
            "avoidTitles": avoid_titles,
            "lovedSoFar": loved_so_far,
            "passedSoFar": passed_so_far,
            "slotsToReplace": batch,
        }
        try:
            text = call_claude(payload)
            last_response = text
            picks = parse_picks(text)
            if not picks:
                raise ValueError("could not parse JSON array from response")
            log.info(f"  batch {bi}/{len(batches)}: parsed {len(picks)} picks")
            all_picks.extend(picks[: len(batch)])
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            log.error(f"× batch {bi}/{len(batches)} failed: {error}")
            log.debug(traceback.format_exc())
            break

    # Apply replacements (only for picks we successfully got)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    replaced: list[dict[str, Any]] = []
    for old_book, pick in zip(marked_books, all_picks):
        m = state["marks"].get(old_book["id"], {})

        # Add old book to history
        state["history"].insert(
            0,
            {
                "id": old_book["id"],
                "title": old_book["title"],
                "author": old_book["author"],
                "genre": old_book["genre"],
                "mark": label_for_mark(m),
                "replaced_at": now,
            },
        )

        # Build replacement
        new_id = make_unique_id(pick.get("title", ""), state)
        new_book = {
            "id": new_id,
            "title": str(pick.get("title", "Untitled")),
            "author": str(pick.get("author", "Unknown")),
            "genre": old_book["genre"],  # force genre to match slot
            "why": str(pick.get("why", "")),
            "is_fresh": True,
        }

        # Replace in place
        for i, b in enumerate(state["current_books"]):
            if b["id"] == old_book["id"]:
                state["current_books"][i] = new_book
                break

        # Clear mark for the old slot id
        state["marks"].pop(old_book["id"], None)
        replaced.append({"old": old_book, "new": new_book})
        log.info(
            f"  replaced [{old_book['genre']}] "
            f"{old_book['title']!r} ({label_for_mark(m)}) → {new_book['title']!r} by {new_book['author']}"
        )

    save_state(state)
    log.info(f"fresh-picks done  replaced={len(replaced)}/{len(marked_books)}  error={error or 'none'}")

    return jsonify(
        {
            "ok": error is None,
            "replaced_count": len(replaced),
            "marked_count": len(marked_books),
            "error": error,
            "last_response_preview": (last_response[:500] if error else None),
            "marks": state["marks"],
            "current_books": state["current_books"],
            "history": state["history"],
        }
    )


@app.route("/api/test", methods=["GET"])
def api_test():
    """Quick connectivity check: minimal Claude call."""
    log.info("test-api  pinging Claude with a 1-token prompt")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"ok": False, "error": "ANTHROPIC_API_KEY not set"}), 500
    try:
        client = Anthropic(api_key=api_key, timeout=20.0)
        t0 = time.time()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        elapsed = time.time() - t0
        text = resp.content[0].text
        log.info(f"test-api OK  latency={elapsed:.1f}s  reply={text!r}")
        return jsonify({"ok": True, "latency_s": round(elapsed, 2), "reply": text, "model": MODEL})
    except Exception as e:
        log.error(f"test-api failed: {type(e).__name__}: {e}")
        log.debug(traceback.format_exc())
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


@app.errorhandler(Exception)
def _handle_exception(e):
    log.error(f"unhandled exception in {request.method} {request.path}: {type(e).__name__}: {e}")
    log.error(traceback.format_exc())
    return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


# ───────────────────── Entry point ─────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Local book recommender server")
    parser.add_argument("--debug", action="store_true", help="Verbose debug logging")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    setup_logging(args.debug)

    log.info("═" * 60)
    log.info("Book recommender server")
    log.info(f"  working dir : {HERE}")
    log.info(f"  state file  : {STATE_FILE}")
    log.info(f"  log file    : {LOG_FILE}")
    log.info(f"  model       : {MODEL}")
    log.info(f"  batch size  : {BATCH_SIZE}")
    log.info(f"  timeout     : {TIMEOUT_S}s per call")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY environment variable is not set.")
        log.error("Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    masked = f"{api_key[:10]}…{api_key[-4:]}" if len(api_key) > 14 else "(short key)"
    log.info(f"  api key     : {masked}")

    log.info(f"  listening on: http://localhost:{args.port}")
    log.info(f"  open this URL in your browser to use the app:")
    log.info(f"     http://localhost:{args.port}")
    log.info("═" * 60)

    try:
        app.run(host="127.0.0.1", port=args.port, debug=args.debug, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e) or getattr(e, "errno", None) == 48:
            log.error(f"Port {args.port} is already in use. Try: python server.py --port {args.port + 1}")
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
