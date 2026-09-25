"""
Local web server for the video game recommender.

Run with:
    python server.py [--debug] [--port 5051] [--model sonnet|haiku|opus] [--user NAME]

Then open http://localhost:5051 in your browser.

Multiple users are supported. Each user has their own profile (favourite
games and developers, genres and seed games) and their own state (current
list, marks, history), stored under users/<name>/. With no --user flag the
server reuses the last user that was run.

There is no default user and no built-in seed data. On the very first run
there are no users yet, so create one before starting the server:

User management (no server is started for these):
    python server.py --list-users        # show all users
    python server.py --new-user NAME      # interactively create a new user
                                          # (asks for favourite games, then
                                          #  Claude builds their starting profile)
    python server.py --new-user NAME --favourites-file games.txt
                                          # read the favourite games from a text
                                          # file (one per line) instead of typing
                                          # them at the prompt

Requires:
    - ANTHROPIC_API_KEY environment variable
    - uv: just run any command as `uv run server.py ...` (deps auto-installed)
    (or pip)
    - pip install -r requirements.txt
    (or anaconda)
    - conda env create -f environment.yml
    - conda activate game_recommender
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

# Local
import steam


# ───────────────────── Configuration ─────────────────────

HERE = Path(__file__).parent.resolve()
USERS_DIR = HERE / "users"
STEAM = steam.SteamCache(HERE / "steam_cache.json")  # shared across users
LAST_USER_FILE = HERE / "last_user.txt"
LOG_FILE = HERE / "server.log"
DEFAULT_PORT = 5051

MODEL = "claude-opus-5"  # overridden by --model at startup

# Set at startup once a user has been resolved. The web routes operate on
# whichever user the server was launched for.
CURRENT_USER = ""
PROFILE: dict[str, Any] = {}

MODELS = {
    "sonnet": "claude-sonnet-5",
    "haiku":  "claude-haiku-4-5",
    "opus":   "claude-opus-5",
}
MAX_TOKENS = 2048
BOOTSTRAP_MAX_TOKENS = 8192   # new-user profile is a much bigger generation
BATCH_SIZE = 6        # games per Claude call
TIMEOUT_S = 90.0      # per-call timeout
BOOTSTRAP_TIMEOUT_S = 180.0   # building a whole profile can take a while


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


# ───────────────────── Users, profiles and state ─────────────────────
#
# Layout on disk:
#     users/<name>/profile.json   favourites, genres, seed_games (the "prompt")
#     users/<name>/state.json     version, current_games, marks, history, also_played
#     last_user.txt               name of the most recently launched user
#
# A profile is the personalised setup for a user. The state is their live
# games list. Keeping them separate means we can regenerate or hand-edit a
# profile without disturbing someone's marks and history.
#
# There is no built-in/default user: every profile is generated by Claude from
# the favourite games the user supplies via `--new-user`.

USER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def normalise_user_name(name: str) -> str:
    """Lowercase and validate a user name; raise ValueError if unusable."""
    cleaned = (name or "").strip().lower()
    if not USER_NAME_RE.match(cleaned):
        raise ValueError(
            f"invalid user name {name!r}: use 1-32 chars of a-z, 0-9, '-' or '_' "
            f"(starting with a letter or digit)"
        )
    return cleaned


def user_dir(user: str) -> Path:
    return USERS_DIR / user


def profile_path(user: str) -> Path:
    return user_dir(user) / "profile.json"


def state_path(user: str) -> Path:
    return user_dir(user) / "state.json"


def user_exists(user: str) -> bool:
    return profile_path(user).exists()


def list_users() -> list[str]:
    if not USERS_DIR.exists():
        return []
    return sorted(d.name for d in USERS_DIR.iterdir() if (d / "profile.json").exists())


def read_last_user() -> str | None:
    try:
        name = LAST_USER_FILE.read_text(encoding="utf-8").strip()
        return name or None
    except FileNotFoundError:
        return None
    except Exception as e:
        log.warning(f"could not read {LAST_USER_FILE.name}: {e}")
        return None


def write_last_user(user: str) -> None:
    try:
        LAST_USER_FILE.write_text(user + "\n", encoding="utf-8")
    except Exception as e:
        log.warning(f"could not record last user: {e}")


# ── Profiles ──

def normalise_profile(profile: dict[str, Any], display_name: str) -> dict[str, Any]:
    """Coerce a (possibly Claude-generated) profile into a valid shape.

    Ensures the required keys exist, genres have keys/labels, and every seed
    game has a unique id and a genre that matches one of the declared genres.
    """
    favs = [str(x) for x in profile.get("original_favourites", []) if str(x).strip()]
    genres = []
    seen_keys: set[str] = set()
    for g in profile.get("genres", []):
        key = re.sub(r"[^a-z0-9]+", "-", str(g.get("key", "")).lower()).strip("-")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        genres.append({
            "key": key,
            "label": str(g.get("label", key)).strip() or key,
            "sub": str(g.get("sub", "")).strip(),
        })

    valid_genre_keys = {g["key"] for g in genres}
    fallback_genre = genres[0]["key"] if genres else "general"

    seed_games = []
    taken_ids: set[str] = set()
    for b in profile.get("seed_games", []):
        title = str(b.get("title", "")).strip()
        if not title:
            continue
        genre = str(b.get("genre", "")).strip()
        if genre not in valid_genre_keys:
            genre = fallback_genre
        base = re.sub(r"[^a-z0-9]+", "-", (b.get("id") or title).lower()).strip("-") or "game"
        bid = base
        n = 2
        while bid in taken_ids:
            bid = f"{base}-{n}"
            n += 1
        taken_ids.add(bid)
        seed_games.append({
            "id": bid,
            "title": title,
            "developer": str(b.get("developer", "Unknown")).strip() or "Unknown",
            "genre": genre,
            "why": str(b.get("why", "")).strip(),
        })

    return {
        "version": 1,
        "display_name": display_name,
        "original_favourites": favs,
        "genres": genres,
        "seed_games": seed_games,
    }


def save_profile(user: str, profile: dict[str, Any]) -> None:
    user_dir(user).mkdir(parents=True, exist_ok=True)
    path = profile_path(user)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    log.debug(
        f"saved profile for {user!r}: {len(profile.get('original_favourites', []))} favourites, "
        f"{len(profile.get('genres', []))} genres, {len(profile.get('seed_games', []))} seed games"
    )


def load_profile(user: str) -> dict[str, Any]:
    path = profile_path(user)
    with path.open("r", encoding="utf-8") as f:
        profile = json.load(f)
    for key in ("original_favourites", "genres", "seed_games"):
        if key not in profile:
            raise ValueError(f"profile for {user!r} is missing key: {key}")
    profile.setdefault("display_name", user)
    return profile


# ── State ──

def default_state(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "current_games": [dict(b) for b in profile.get("seed_games", [])],
        "marks": {},
        "history": [],
        "also_played": [],
    }


def load_state(user: str, profile: dict[str, Any]) -> dict[str, Any]:
    path = state_path(user)
    if not path.exists():
        log.info(f"No state file for {user!r} — creating fresh state from profile")
        st = default_state(profile)
        save_state(user, st)
        return st

    try:
        with path.open("r", encoding="utf-8") as f:
            st = json.load(f)
        for key in ("current_games", "marks", "history"):
            if key not in st:
                raise ValueError(f"missing key: {key}")
        # Added later; older state files won't have it
        st.setdefault("also_played", [])
        return st
    except Exception as e:
        backup = path.with_suffix(f".corrupt-{int(time.time())}.json")
        log.error(f"State file for {user!r} corrupt ({e}). Backing up to {backup.name} and starting fresh.")
        try:
            path.rename(backup)
        except Exception as ee:
            log.error(f"Could not rename corrupt state file: {ee}")
        st = default_state(profile)
        save_state(user, st)
        return st


def save_state(user: str, state: dict[str, Any]) -> None:
    user_dir(user).mkdir(parents=True, exist_ok=True)
    path = state_path(user)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    log.debug(
        f"saved state for {user!r}: {len(state['current_games'])} games, "
        f"{len(state['marks'])} marks, {len(state['history'])} history"
    )


# ───────────────────── Claude integration ─────────────────────

SYSTEM_PROMPT = """You recommend video games. The user gives you a list of their original favourite games and developers, a list of titles to avoid, and a list of "slots to replace" — each slot has a genre and a mark indicating how the user reacted to the game it's replacing. For each slot, return one new recommendation.

"lovedSoFar" lists games the user has loved in this app so far (including in earlier rounds) — treat them like extra favourites and let them drive your picks as much as originalFavourites do. "passedSoFar" lists games they did not take to — steer away from that style. "alsoPlayed" lists games played outside this app with the user's reaction; the loved/passed ones are already in those two lists. avoidTitles lists other games the user has already seen or played (no strong reaction). Never recommend anything in lovedSoFar, passedSoFar, alsoPlayed or avoidTitles.

Output format: your entire reply must be a single JSON array. The first character must be [ and the last must be ]. No prose, no preamble, no code fences, no commentary.

Each element must be an object with exactly these fields:
{"title": string, "developer": string, "genre": string, "why": string}

The "genre" field must equal the slot's genre exactly. The "developer" field is the studio or creator. The "why" field is 1-2 short sentences tying the pick to a specific original favourite or to the game being replaced.

Rules per replacement:
- "genre" must equal the slot's genre exactly
- The game must NOT appear in originalFavourites, lovedSoFar, passedSoFar, alsoPlayed or avoidTitles
- mark = "loved": pick something stylistically adjacent (same vibe, mechanics, themes)
- mark = "passed": pick something in the same genre but with a clearly different style or approach
- mark = "played": pick a strong adjacent game that broadens exposure
- Lesser-known, indie, and older picks are welcome
"""


BOOTSTRAP_SYSTEM_PROMPT = """You set up a personalised video game recommender for a new user. You are given the user's favourite games and developers, and optionally a hint about the kinds of genres they enjoy. Build their starting profile.

Output format: your entire reply must be a single JSON object. The first character must be { and the last must be }. No prose, no preamble, no code fences, no commentary.

The object must have exactly these fields:
{
  "original_favourites": [string, ...],
  "genres": [ {"key": string, "label": string, "sub": string}, ... ],
  "seed_games": [ {"id": string, "title": string, "developer": string, "genre": string, "why": string}, ... ]
}

Requirements:
- "original_favourites": 10-16 entries, each a tidied "Title — Developer or note" line derived from what the user told you. Keep their own games here.
- "genres": exactly 6. "key" is a short lowercase-kebab slug (e.g. "soulslike", "cozy-sim"). "label" is a human title (e.g. "Soulslike action-RPGs"). "sub" is a short hint like "For when you want X", referencing one of the user's favourites. Genres must reflect the user's stated tastes (and the genre hint if given).
- "seed_games": exactly 24 — exactly 4 per genre. "genre" must equal one of the genre keys above. "id" is a unique lowercase-kebab slug. "why" is 1-2 sentences tying the pick to a specific favourite game/developer or to the genre.
- seed_games must NOT include any game already listed in original_favourites.
- Lesser-known, indie, and older picks are welcome.
"""


def call_claude(
    payload: dict[str, Any],
    *,
    system: str = SYSTEM_PROMPT,
    max_tokens: int = MAX_TOKENS,
    timeout: float = TIMEOUT_S,
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    client = Anthropic(api_key=api_key, timeout=timeout)
    user_content = json.dumps(payload, indent=2, ensure_ascii=False)

    log.info(
        f"→ Claude  model={MODEL}  prompt_chars={len(user_content)}  "
        f"slots={len(payload.get('slotsToReplace', []))}"
    )
    log.info("── system prompt ──\n" + system)
    log.info("── user prompt ──\n" + user_content)

    t0 = time.time()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
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
    text = _response_text(response)
    usage = getattr(response, "usage", None)
    usage_str = ""
    if usage is not None:
        usage_str = f"  in_tok={getattr(usage, 'input_tokens', '?')}  out_tok={getattr(usage, 'output_tokens', '?')}"
    log.info(f"← Claude  latency={elapsed:.1f}s  resp_chars={len(text)}{usage_str}")
    log.debug(f"raw response:\n{text}")

    return text


def _response_text(response: Any) -> str:
    """Join the text blocks of a response.

    Claude 5 models think by default, so response.content may start with a
    ThinkingBlock (no .text) — never assume content[0] is the text block.
    """
    return "".join(b.text for b in response.content if b.type == "text")


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
    taken = {b["id"] for b in state["current_games"]} | {h.get("id", "") for h in state["history"]}
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
    return "played"


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Forgiving parser for a single JSON object (used for new-user profiles)."""
    s = text.strip()
    s = re.sub(r"^```(?:json|javascript|js)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s, flags=re.IGNORECASE).strip()

    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as e:
        log.debug(f"direct object parse failed: {e}")

    start = s.find("{")
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
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError as e:
                    log.debug(f"balanced object parse failed: {e}")
                    return None
    return None


# ───────────────────── New-user creation (CLI) ─────────────────────

def _prompt_multiline(prompt: str) -> str:
    """Read lines from stdin until a blank line or EOF."""
    print(prompt)
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def read_favourites_file(path: str) -> str:
    """Read a favourites text file: strip blank lines, keep one game per line."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        print(f"ERROR: could not read favourites file {path!r}: {e}", file=sys.stderr)
        sys.exit(1)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return "\n".join(lines)


def generate_profile(favourites_text: str, genre_hint: str, display_name: str) -> dict[str, Any]:
    """Ask Claude to build a starting profile from the user's stated tastes."""
    payload: dict[str, Any] = {"favourites": favourites_text}
    if genre_hint:
        payload["genreHint"] = genre_hint

    text = call_claude(
        payload,
        system=BOOTSTRAP_SYSTEM_PROMPT,
        max_tokens=BOOTSTRAP_MAX_TOKENS,
        timeout=BOOTSTRAP_TIMEOUT_S,
    )
    raw = parse_json_object(text)
    if not raw:
        raise ValueError(
            "could not parse a JSON profile from Claude's response.\n"
            f"First 500 chars of response:\n{text[:500]}"
        )
    profile = normalise_profile(raw, display_name)
    if not profile["genres"] or not profile["seed_games"]:
        raise ValueError("Claude returned a profile with no usable genres or seed games")
    return profile


def create_user_interactive(name: str, favourites_file: str | None = None) -> None:
    """Interactively create a new user: collect tastes, generate a profile.

    If favourites_file is given, the favourite games are read from that file
    (one per line) instead of being typed at the prompt.
    """
    try:
        user = normalise_user_name(name)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    if user_exists(user):
        print(f"User {user!r} already exists ({profile_path(user)}). Choose another name.", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set — needed to build the new user's profile.", file=sys.stderr)
        sys.exit(1)

    print("═" * 60)
    print(f"Creating a new game-recommender user: {user!r}")
    print("═" * 60)
    display_name = input("Display name (press Enter to use the user name): ").strip() or user

    if favourites_file is not None:
        favourites_text = read_favourites_file(favourites_file)
        n_lines = len(favourites_text.splitlines())
        print(f"\nRead {n_lines} favourite(s) from {favourites_file}")
    else:
        favourites_text = _prompt_multiline(
            "\nList this person's favourite video games and developers — one per line, free-form.\n"
            "Example:\n"
            "  Hollow Knight — Team Cherry\n"
            "  The Witcher 3, especially the side quests\n"
            "Finish with a blank line:"
        )
    if not favourites_text:
        print("No favourites given — cannot build a profile. Aborting.", file=sys.stderr)
        sys.exit(1)

    genre_hint = input(
        "\n(Optional) Any genres to emphasise? e.g. 'metroidvania, crpg, roguelike'.\n"
        "Press Enter to let Claude choose: "
    ).strip()

    print(f"\nAsking Claude (model {MODEL}) to build a starting profile — this can take a minute…")
    try:
        profile = generate_profile(favourites_text, genre_hint, display_name)
    except Exception as e:
        print(f"\nFailed to build profile: {e}", file=sys.stderr)
        sys.exit(1)

    save_profile(user, profile)
    save_state(user, default_state(profile))
    write_last_user(user)

    print("\n" + "─" * 60)
    print(f"Created user {user!r} ({display_name}):")
    print(f"  {len(profile['original_favourites'])} favourites")
    print(f"  {len(profile['genres'])} genres: " + ", ".join(g["label"] for g in profile["genres"]))
    print(f"  {len(profile['seed_games'])} seed games")
    print(f"  profile : {profile_path(user)}")
    print(f"  state   : {state_path(user)}")
    print("─" * 60)
    print(f"\nStart the server for this user with:\n    python server.py --user {user}")
    print("(Edit the profile.json by hand any time to fine-tune the favourites, genres or seed games.)")


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
    state = load_state(CURRENT_USER, PROFILE)
    return jsonify(
        {
            "current_games": state["current_games"],
            "marks": state["marks"],
            "history": state["history"],
            "also_played": state["also_played"],
            "genres": PROFILE["genres"],
            "original_favourites": PROFILE["original_favourites"],
            "model": MODEL,
            "user": CURRENT_USER,
            "display_name": PROFILE.get("display_name", CURRENT_USER),
            "users": list_users(),
        }
    )


@app.route("/api/users", methods=["GET"])
def api_users():
    return jsonify({
        "users": list_users(),
        "current": CURRENT_USER,
        "display_name": PROFILE.get("display_name", CURRENT_USER),
    })


@app.route("/api/mark", methods=["POST"])
def api_mark():
    body = request.get_json(silent=True) or {}
    game_id = body.get("id")
    patch = body.get("patch", {}) or {}
    if not game_id or not isinstance(patch, dict):
        return jsonify({"ok": False, "error": "Missing id or patch"}), 400

    log.info(f"mark  user={CURRENT_USER!r}  id={game_id!r}  patch={patch}")

    state = load_state(CURRENT_USER, PROFILE)
    cur = state["marks"].get(game_id, {"played": False, "like": None})
    cur = {**cur, **patch}
    if patch.get("like") in ("love", "meh"):
        cur["played"] = True
    state["marks"][game_id] = cur
    save_state(CURRENT_USER, state)
    return jsonify({"ok": True, "marks": state["marks"]})


@app.route("/api/clear-marks", methods=["POST"])
def api_clear_marks():
    log.info(f"clear all marks  user={CURRENT_USER!r}")
    state = load_state(CURRENT_USER, PROFILE)
    state["marks"] = {}
    save_state(CURRENT_USER, state)
    return jsonify({"ok": True, "marks": state["marks"]})


@app.route("/api/also-played", methods=["POST"])
def api_also_played_add():
    """Record a game played outside the app: {title, like: "love"|"meh"|null}."""
    body = request.get_json(silent=True) or {}
    title = str(body.get("title") or "").strip()
    like = body.get("like")
    if not title:
        return jsonify({"ok": False, "error": "Missing title"}), 400
    if like not in ("love", "meh", None):
        return jsonify({"ok": False, "error": "like must be 'love', 'meh' or null"}), 400

    log.info(f"also-played add  user={CURRENT_USER!r}  title={title!r}  like={like}")
    state = load_state(CURRENT_USER, PROFILE)
    # Same title again (any case) replaces the earlier entry
    state["also_played"] = [e for e in state["also_played"] if e["title"].lower() != title.lower()]
    state["also_played"].append({"title": title, "like": like, "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    save_state(CURRENT_USER, state)
    return jsonify({"ok": True, "also_played": state["also_played"]})


@app.route("/api/also-played/remove", methods=["POST"])
def api_also_played_remove():
    body = request.get_json(silent=True) or {}
    title = str(body.get("title") or "").strip().lower()
    if not title:
        return jsonify({"ok": False, "error": "Missing title"}), 400
    log.info(f"also-played remove  user={CURRENT_USER!r}  title={title!r}")
    state = load_state(CURRENT_USER, PROFILE)
    state["also_played"] = [e for e in state["also_played"] if e["title"].lower() != title]
    save_state(CURRENT_USER, state)
    return jsonify({"ok": True, "also_played": state["also_played"]})


@app.route("/api/clear-history", methods=["POST"])
def api_clear_history():
    log.info(f"clear history  user={CURRENT_USER!r}")
    state = load_state(CURRENT_USER, PROFILE)
    state["history"] = []
    save_state(CURRENT_USER, state)
    return jsonify({"ok": True, "history": state["history"]})


@app.route("/api/fresh-picks", methods=["POST"])
def api_fresh_picks():
    state = load_state(CURRENT_USER, PROFILE)
    original_favourites = PROFILE["original_favourites"]
    marked_games = [
        b for b in state["current_games"]
        if (m := state["marks"].get(b["id"])) and (m.get("played") or m.get("like"))
    ]
    if not marked_games:
        log.warning("fresh-picks called with no marked games")
        return jsonify({"ok": False, "error": "No marked games to replace."}), 400

    log.info(f"fresh-picks  marked={len(marked_games)}")

    also_played = [
        {"title": e["title"], "reaction": label_for_mark({"played": True, "like": e.get("like")})}
        for e in state["also_played"]
    ]
    # Taste signal accumulates across rounds: games on the board now, games
    # replaced in earlier rounds (history keeps their mark), and alsoPlayed.
    def with_reaction(reaction: str) -> list[str]:
        return (
            [
                f"{b['title']} by {b['developer']}"
                for b in state["current_games"]
                if label_for_mark(state["marks"].get(b["id"], {})) == reaction
                and state["marks"].get(b["id"], {}).get("like")
            ]
            + [f"{h['title']} by {h['developer']}" for h in state["history"] if h.get("mark") == reaction]
            + [e["title"] for e in state["also_played"] if e.get("like") and label_for_mark(e) == reaction]
        )

    loved_so_far = with_reaction("loved")
    passed_so_far = with_reaction("passed")
    # Loved/passed games are already listed above; keep them out of avoidTitles
    # so the model doesn't read a loved game as something to steer clear of.
    already_listed = {t.casefold() for t in loved_so_far + passed_so_far}
    avoid_titles = [
        t for t in (
            [f"{b['title']} by {b['developer']}" for b in state["current_games"]]
            + [f"{h['title']} by {h['developer']}" for h in state["history"]]
            + [e["title"] for e in state["also_played"]]
        )
        if t.casefold() not in already_listed
    ]

    slots = []
    for b in marked_games:
        m = state["marks"][b["id"]]
        slots.append(
            {
                "genre": b["genre"],
                "replacing": f"{b['title']} by {b['developer']}",
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
            "originalFavourites": original_favourites,
            "avoidTitles": avoid_titles,
            "lovedSoFar": loved_so_far,
            "passedSoFar": passed_so_far,
            "alsoPlayed": also_played,
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
    for old_game, pick in zip(marked_games, all_picks):
        m = state["marks"].get(old_game["id"], {})

        # Add old game to history
        state["history"].insert(
            0,
            {
                "id": old_game["id"],
                "title": old_game["title"],
                "developer": old_game["developer"],
                "genre": old_game["genre"],
                "mark": label_for_mark(m),
                "replaced_at": now,
            },
        )

        # Build replacement
        new_id = make_unique_id(pick.get("title", ""), state)
        new_game = {
            "id": new_id,
            "title": str(pick.get("title", "Untitled")),
            "developer": str(pick.get("developer", "Unknown")),
            "genre": old_game["genre"],  # force genre to match slot
            "why": str(pick.get("why", "")),
            "is_fresh": True,
        }

        # Replace in place
        for i, b in enumerate(state["current_games"]):
            if b["id"] == old_game["id"]:
                state["current_games"][i] = new_game
                break

        # Clear mark for the old slot id
        state["marks"].pop(old_game["id"], None)
        replaced.append({"old": old_game, "new": new_game})
        log.info(
            f"  replaced [{old_game['genre']}] "
            f"{old_game['title']!r} ({label_for_mark(m)}) → {new_game['title']!r} by {new_game['developer']}"
        )

    save_state(CURRENT_USER, state)
    log.info(f"fresh-picks done  user={CURRENT_USER!r}  replaced={len(replaced)}/{len(marked_games)}  error={error or 'none'}")

    return jsonify(
        {
            "ok": error is None,
            "replaced_count": len(replaced),
            "marked_count": len(marked_games),
            "error": error,
            "last_response_preview": (last_response[:500] if error else None),
            "marks": state["marks"],
            "current_games": state["current_games"],
            "history": state["history"],
        }
    )


@app.route("/api/steam", methods=["GET"])
def api_steam():
    """Steam review rating for a title (cached). rating is null when unmatched."""
    title = (request.args.get("title") or "").strip()
    if not title:
        return jsonify({"ok": False, "error": "Missing title"}), 400
    return jsonify({"ok": True, "rating": STEAM.get(title)})


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
        text = _response_text(resp)
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

def resolve_user(requested: str | None) -> str:
    """Decide which user to run as.

    Precedence: explicit --user, then the last-used user, then the sole user if
    there is exactly one. There is no built-in default user, so a fresh install
    with no users exits with a helpful message telling you to create one.
    """
    available = list_users()

    if requested is not None:
        try:
            user = normalise_user_name(requested)
        except ValueError as e:
            log.error(str(e))
            sys.exit(1)
        if not user_exists(user):
            log.error(f"No such user: {user!r}.")
            if available:
                log.error(f"Available users: {', '.join(available)}")
            log.error(f"Create one with:  python server.py --new-user {user}")
            sys.exit(1)
        return user

    last = read_last_user()
    if last and user_exists(last):
        return last

    if len(available) == 1:
        return available[0]

    log.error("No user could be selected.")
    if available:
        log.error(f"Pick one with --user. Available users: {', '.join(available)}")
    else:
        log.error("There are no users yet — this is probably your first run.")
        log.error("Create one with:  python server.py --new-user NAME")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local video game recommender server")
    parser.add_argument("--debug", action="store_true", help="Verbose debug logging")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", choices=["sonnet", "haiku", "opus"], default="opus",
                        help="Claude model to use (default: opus)")
    parser.add_argument("--user", default=None,
                        help="Which user to run as (default: the last user)")
    parser.add_argument("--new-user", metavar="NAME", default=None,
                        help="Interactively create a new user, then exit")
    parser.add_argument("--favourites-file", metavar="PATH", default=None,
                        help="With --new-user: read the favourite games from this "
                             "text file (one per line) instead of prompting")
    parser.add_argument("--list-users", action="store_true",
                        help="List all users, then exit")
    args = parser.parse_args()

    global MODEL, CURRENT_USER, PROFILE
    MODEL = MODELS[args.model]

    setup_logging(args.debug)

    # ── User-management sub-commands (no server) ──
    if args.list_users:
        users = list_users()
        last = read_last_user()
        if not users:
            print("No users yet. Create one with:  python server.py --new-user NAME")
        else:
            print("Users:")
            for u in users:
                marks = []
                if u == last:
                    marks.append("last used")
                suffix = f"  ({', '.join(marks)})" if marks else ""
                print(f"  {u}{suffix}")
        return

    if args.favourites_file is not None and args.new_user is None:
        parser.error("--favourites-file only makes sense with --new-user")

    if args.new_user is not None:
        create_user_interactive(args.new_user, favourites_file=args.favourites_file)
        return

    # ── Resolve and load the user we'll serve ──
    CURRENT_USER = resolve_user(args.user)
    try:
        PROFILE = load_profile(CURRENT_USER)
    except Exception as e:
        log.error(f"Could not load profile for {CURRENT_USER!r}: {e}")
        sys.exit(1)
    write_last_user(CURRENT_USER)

    log.info("═" * 60)
    log.info("Video game recommender server")
    log.info(f"  working dir : {HERE}")
    log.info(f"  user        : {CURRENT_USER}  ({PROFILE.get('display_name', CURRENT_USER)})")
    log.info(f"  profile     : {profile_path(CURRENT_USER)}")
    log.info(f"  state file  : {state_path(CURRENT_USER)}")
    log.info(f"  log file    : {LOG_FILE}")
    log.info(f"  model       : {MODEL}")
    log.info(f"  batch size  : {BATCH_SIZE}")
    log.info(f"  timeout     : {TIMEOUT_S}s per call")
    other_users = [u for u in list_users() if u != CURRENT_USER]
    if other_users:
        log.info(f"  other users : {', '.join(other_users)}  (switch with --user)")

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
