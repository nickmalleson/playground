# Build spec: a personal book recommender (Python CLI)

I want you to build me a terminal-based personal book recommender in Python. I previously tried this as a browser-based artifact, but the in-sandbox API for calling Claude was flaky and hard to debug, so I'm moving to a CLI that talks to the Anthropic API directly.

Read this whole document before writing code. Ask me questions only if something is genuinely ambiguous; otherwise just build it.

---

## What it does

1. Shows me a curated set of book recommendations, grouped by 6 genres.
2. Lets me mark each book as **read**, **loved**, or **passed**.
3. Persists my marks across runs in a JSON file in the same directory
4. Within each genre, surfaces loved books to the top and sinks passed/read ones.
5. Re-orders the genre sections themselves so genres I've loved more rise.
6. When I ask for "fresh picks", calls the Anthropic API to generate ONE tailored replacement per marked book — same genre slot, tailored to whether I loved/passed/read it. Marked books then move into a history list.
7. Keeps a permanent history of every book I've ever marked, even after replacement.

---

## Tech stack

- Python 3.10+
- `anthropic` — install with `pip install anthropic`
- `rich` — install with `pip install rich`
- Standard library only otherwise (json, pathlib, argparse, dataclasses, datetime, typing)
- Create a new anaconda environment called book-recommender

Read the API key from the `ANTHROPIC_API_KEY` environment variable. If it's unset, exit with a clear message telling me to do `export ANTHROPIC_API_KEY=...`.

Project layout: small, debuggable. A single `recommender.py` is fine, or split into `recommender.py`, `data.py`, `state.py`, `claude.py` — your call. Prefer simplicity.

---

## CLI design

Use `rich` for tables, panels, and colour. Use plain `input()` for prompts. Don't pull in a heavy TUI framework like Textual — keep the loop synchronous and printable.

### Top-level screen

```
══════════════ Your book recommender ══════════════

  Marked: 3    Loved: 1    Passed: 1    In history: 0

  Genres
    1. Climate dystopia          4 books · 1 marked
    2. Cyberpunk and tech-noir   4 books
    3. Space opera & hard SF     4 books · 2 marked
    4. Epic fantasy              4 books
    5. Historical fiction        4 books
    6. Literary speculative      4 books

  Commands
    [number]   open a genre
    g          generate fresh picks (replace 3 marked books)
    h          show marked-books history
    s          show stats
    r          reset all marks (history kept)
    q          quit

>
```

The "g" command's hint should reflect the current count: "(nothing marked yet)" if 0, otherwise "(replace N marked books)".

### Genre screen

```
══════════════ Climate dystopia ══════════════
For when you want Bacigalupi, Atwood or McCarthy

  [1]  Parable of the Sower — Octavia E. Butler
       Walled communities, climate collapse, a teenage prophet on the road. The
       bone-deep dread of The Road meets the political thrust of Bacigalupi.

  [2]  (LOVED) The Ministry for the Future — Kim Stanley Robinson
       A polyphonic climate novel that opens with a heatwave scene as
       devastating as anything in The Water Knife…

  [3]  American War — Omar El Akkad
       A second US civil war in the 2070s, fought over the last fossil fuels…

  [4]  (PASSED) Gold Fame Citrus — Claire Vaye Watkins
       California's water is gone and a dune sea is swallowing the Southwest…

  Commands
    [number]   mark a book
    b          back
    g          generate fresh picks

>
```

Use colour for the status tags: green for LOVED, dim/grey for PASSED and READ, slight dim for the whole row when read or passed. Loved rows can have a subtle border or accent.

### Mark screen

```
"Parable of the Sower" by Octavia E. Butler

  l   loved it
  p   not for me
  r   mark read (no strong opinion)
  u   unmark
  c   cancel

>
```

After marking, return to the same genre screen automatically. If a book is marked loved or passed, also set its read state to true.

### Generate fresh picks

When I press `g`:

```
3 books are marked. Asking Claude for tailored replacements…

  Batch 1 of 1 (3 replacements)… done in 11.7s

  ✓ Climate dystopia
    REPLACED: The Ministry for the Future  (loved)
    WITH:     The Overstory — Richard Powers
              "Sprawling ecological epic that scales like KSR but with Mitchell's
              polyphony — your loved book pointed straight at this."

  ✓ Space opera & hard SF
    REPLACED: A Memory Called Empire  (passed)
    WITH:     Embassytown — China Miéville
              "Court intrigue is out; you've signalled you want weirder, more
              alien-language driven space fic. Miéville delivers."

  ✓ Cyberpunk and tech-noir
    REPLACED: Snow Crash  (read)
    WITH:     Synners — Pat Cadigan
              "A criminally underread cyberpunk novel from the same Gibson era,
              broadens what you've already sampled."

3 replaced, 0 failed. Returning to main menu.
```

If a batch fails, commit the successful ones and report which failed.

### History screen

A simple paginated list, newest first:

```
══════════════ Marked books history (12) ══════════════

  LOVED   The Ministry for the Future — Kim Stanley Robinson
          climate dystopia · replaced 3 days ago

  PASSED  A Memory Called Empire — Arkady Martine
          space opera · replaced 3 days ago

  READ    Snow Crash — Neal Stephenson
          cyberpunk · replaced 3 days ago

  …

  Commands: b back · c clear history · q quit
```

---

## Data model

```python
from dataclasses import dataclass, asdict
from typing import Optional, Literal

GenreKey = Literal["cli-fi", "cyber", "space", "fantasy", "historical", "literary"]
MarkLike = Literal["love", "meh"]
MarkLabel = Literal["loved", "passed", "read"]

@dataclass
class Book:
    id: str
    title: str
    author: str
    genre: GenreKey
    why: str
    is_fresh: bool = False  # True for replacements generated by Claude

@dataclass
class Mark:
    read: bool = False
    like: Optional[MarkLike] = None  # "love" | "meh" | None

@dataclass
class HistoryEntry:
    title: str
    author: str
    genre: GenreKey
    mark: MarkLabel  # "loved" | "passed" | "read"
    replaced_at: str  # ISO 8601 timestamp
```

### State file

Path: `~/.book_recommender/state.json`. Created on first run. Shape:

```json
{
  "version": 1,
  "current_books": [
    {"id": "parable-of-the-sower", "title": "Parable of the Sower",
     "author": "Octavia E. Butler", "genre": "cli-fi",
     "why": "...", "is_fresh": false},
    ...
  ],
  "marks": {
    "parable-of-the-sower": {"read": true, "like": "love"}
  },
  "history": [
    {"title": "...", "author": "...", "genre": "cli-fi",
     "mark": "loved", "replaced_at": "2026-05-03T12:34:56Z"},
    ...
  ]
}
```

On first run, seed `current_books` from `SEED_BOOKS` (defined below), with empty `marks` and `history`. Save after every state change. Use atomic writes (write to temp file, rename) so a crash mid-write doesn't corrupt state.

### Genre score (used for re-ranking)

```python
def genre_score(genre_key: GenreKey, books, marks) -> float:
    score = 0.0
    for b in books:
        if b.genre != genre_key:
            continue
        m = marks.get(b.id) or Mark()
        if m.like == "love":  score += 2.0
        if m.like == "meh":   score -= 2.0
        if m.read and m.like is None:  score += 0.25
    return score
```

Within a genre, sort books by: loved → unmarked → read-only → passed.

---

## Claude API integration

Use the `anthropic` SDK, model `claude-sonnet-4-6`. (Don't fall back to Haiku — quality matters here, and the previous attempt with Haiku produced poor recommendations.)

```python
from anthropic import Anthropic

client = Anthropic()  # picks up ANTHROPIC_API_KEY

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_payload}],
)
text = response.content[0].text
```

### System prompt (use verbatim)

```
You recommend books. The user gives you a list of their original favourite books and authors, a list of titles to avoid, and a list of "slots to replace" — each slot has a genre and a mark indicating how the user reacted to the book it's replacing. For each slot, return one new recommendation.

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
```

### User payload

A JSON-stringified object:

```python
user_payload = json.dumps({
    "originalFavourites": ORIGINAL_FAVOURITES,
    "avoidTitles": [f"{b.title} by {b.author}" for b in current_books]
                 + [f"{h.title} by {h.author}" for h in history],
    "lovedSoFar": [...],
    "passedSoFar": [...],
    "slotsToReplace": [
        {"genre": b.genre, "replacing": f"{b.title} by {b.author}", "mark": mark_label}
        for b in marked_books_in_this_batch
    ],
}, indent=2)
```

### Batching

If more than 6 books are marked, split into batches of 6. Process sequentially. Commit each batch's successful results to disk BEFORE starting the next batch. That way a crash or rate limit halfway through doesn't lose anything.

Per-call timeout: 90 seconds. If a batch fails, log the error, leave that batch's books unmodified, and continue or stop based on the failure type (skip on parse failure; stop on auth or rate-limit error).

### Response parsing (be forgiving)

1. Trim whitespace.
2. Strip markdown code fences if present (e.g. `\`\`\`json … \`\`\``).
3. Try `json.loads` on the cleaned text.
4. If parse returns a dict (Claude wrapped the array in `{"recommendations": [...]}` or similar), grab the first list-valued property.
5. If parse fails, find the first balanced `[…]` block (track string state and bracket depth) and try parsing that.
6. If still no array, log the raw response and abort the batch.

For each pick: validate it has title, author, genre, why fields and that genre matches the slot. If genre doesn't match the slot's genre, force it to match (don't reject — keep the slot in its genre).

### Replacement step (after parse)

For each `(marked_book, pick)` pair:
1. Append a HistoryEntry for the marked book to `history` (newest first).
2. Build a new Book with `is_fresh=True`, a fresh slug-based ID (collision-safe — append a number if needed), genre forced to the slot's genre.
3. Replace the old book in `current_books` (preserving position).
4. Delete `marks[old_book.id]`.
5. Save state.

---

## Debug features (this is important — I want this debuggable)

- `--verbose` / `-v` — print every API call's model, payload size in chars, response size, latency, and the first 1000 chars of the raw response.
- `-vv` — full raw response, no truncation.
- `--dry-run` — don't call the API. Generate placeholder replacements like `Book(title="Placeholder for {old.title}", ...)` so I can test marking, state, history, and UI logic in isolation.
- `--test-api` — make one minimal call (just "say hi" with `max_tokens=20`), print model, response text, latency, and exit. Good for verifying the API key and connectivity.
- All API calls log to `~/.book_recommender/log.jsonl`, append-only, one JSON object per line: `{ts, model, prompt_chars, response_chars, latency_ms, error?}`. Don't log the full prompt/response by default — that bloats the log; do log them when `-v` is on.

---

## Error handling

Be loud and useful, never silent.

- Missing `ANTHROPIC_API_KEY`: exit with instructions.
- Network/auth errors: print the error class and message, leave state untouched, return to menu.
- Rate-limit (429): tell me what to do (wait N seconds, retry).
- Parse failure: print the first 500 chars of the raw response with a clear "couldn't parse this — leaving state unchanged" message.
- 90s timeout per batch: handle as a normal failure, don't crash.
- Corrupt state file on load: back it up to `state.json.corrupt-{ts}` and start fresh.

Crashes should always print a traceback to stderr. Don't swallow exceptions.

---

## Seed data

Use this exactly. Don't paraphrase or rewrite the descriptions.

### `ORIGINAL_FAVOURITES`

```python
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
```

### `GENRES`

```python
GENRES = [
    {"key": "cli-fi",     "label": "Climate dystopia",            "sub": "For when you want Bacigalupi, Atwood or McCarthy"},
    {"key": "cyber",      "label": "Cyberpunk and tech-noir",     "sub": "For when you want Gibson"},
    {"key": "space",      "label": "Space opera and hard sci-fi", "sub": "For when you want Hyperion, Three Body or Dune"},
    {"key": "fantasy",    "label": "Epic fantasy",                "sub": "For when you want Abercrombie, GRRM or Lynch"},
    {"key": "historical", "label": "Historical fiction",          "sub": "For when you want Cornwell or Harris"},
    {"key": "literary",   "label": "Literary speculative",        "sub": "For when you want Le Guin or Mitchell"},
]
```

### `SEED_BOOKS`

```python
SEED_BOOKS = [
    # Climate dystopia
    Book("parable-of-the-sower", "Parable of the Sower", "Octavia E. Butler", "cli-fi",
         "Walled communities, climate collapse, a teenage prophet on the road. The bone-deep dread of The Road meets the political thrust of Bacigalupi — and it was written in 1993."),
    Book("ministry-future", "The Ministry for the Future", "Kim Stanley Robinson", "cli-fi",
         "A polyphonic climate novel that opens with a heatwave scene as devastating as anything in The Water Knife, then pivots into oddly hopeful policy wonkery."),
    Book("american-war", "American War", "Omar El Akkad", "cli-fi",
         "A second US civil war in the 2070s, fought over the last fossil fuels. Spare, devastating prose — distinctly McCarthy-coded, with a slow-burn protagonist arc."),
    Book("gold-fame-citrus", "Gold Fame Citrus", "Claire Vaye Watkins", "cli-fi",
         "California's water is gone and a dune sea is swallowing the Southwest. Lyrical and hallucinatory — basically a literary sibling to The Water Knife."),

    # Cyberpunk
    Book("snow-crash", "Snow Crash", "Neal Stephenson", "cyber",
         "Cyberpunk's other foundational text — funnier and faster than Gibson. Samurai pizza couriers, a metaverse linguistic virus, and the best opening 50 pages in the genre."),
    Book("altered-carbon", "Altered Carbon", "Richard Morgan", "cyber",
         "Hard-boiled detective noir plus downloadable consciousness. The Sprawl trilogy's 2000s grandchild — grittier, pulpier, and very confident."),
    Book("quantum-thief", "The Quantum Thief", "Hannu Rajaniemi", "cyber",
         "A post-cyberpunk caper across a transhuman solar system. Dense, dazzling, throws you in the deep end exactly the way Neuromancer did."),
    Book("blindsight", "Blindsight", "Peter Watts", "cyber",
         "First-contact horror with transhumans and predator-vampires interrogating consciousness itself. Brutally smart — Hyperion fans tend to love it too."),

    # Space opera & hard SF
    Book("memory-called-empire", "A Memory Called Empire", "Arkady Martine", "space",
         "A small-station ambassador navigates the seductive horror of a vast empire. Reads like Hyperion crossed with Le Guin's anthropological eye. The sequel is just as good."),
    Book("player-of-games", "The Player of Games", "Iain M. Banks", "space",
         "The friendliest doorway into the Culture. Scope, civilisational politics, and moral weight that should hit your Dune and Hyperion nerve. Banks is the missing author from your list."),
    Book("children-of-time", "Children of Time", "Adrian Tchaikovsky", "space",
         "Uplifted spiders evolve over millennia while a battered generation ship limps toward them. Big-idea SF in Cixin Liu's lineage but with more empathy."),
    Book("house-of-suns", "House of Suns", "Alastair Reynolds", "space",
         "A six-million-year galactic timescale, with clones of a single woman trading memories at meet-ups every 200,000 years. Hyperion-grade scope and melancholy."),

    # Epic fantasy
    Book("gardens-of-the-moon", "Gardens of the Moon (Malazan)", "Steven Erikson", "fantasy",
         "An army-scale, mythologically dense fantasy spread across continents and millennia. Steeper learning curve than First Law but the payoff over ten books is enormous."),
    Book("black-company", "The Black Company", "Glen Cook", "fantasy",
         "Grimdark before grimdark was a word. Mercenary-company POV, blood-and-mud morality — the obvious ancestor of Abercrombie's First Law."),
    Book("assassins-apprentice", "Assassin's Apprentice", "Robin Hobb", "fantasy",
         "Slow-burn, ferocious character work across a 16-book braided saga. The author Rothfuss readers most often graduate to — and she actually finishes her series."),
    Book("poppy-war", "The Poppy War", "R. F. Kuang", "fantasy",
         "Military academy to genocide-scale war, drawn from 20th-century Chinese history. Brutal, modern, GRRM-territory stakes — not for the faint-hearted."),

    # Historical fiction
    Book("wolf-hall", "Wolf Hall", "Hilary Mantel", "historical",
         "Thomas Cromwell's rise in Henry VIII's court. Prose good enough to be illegal, plus all the political cunning of GRRM but in actual history. The trilogy is one of the great achievements."),
    Book("shogun", "Shōgun", "James Clavell", "historical",
         "An Englishman shipwrecked into Sengoku-era Japan. Massive, addictive, written with the same propulsive instinct as Cornwell's Saxon books."),
    Book("master-and-commander", "Master and Commander", "Patrick O'Brian", "historical",
         "Napoleonic naval fiction across 20 books. The more literary cousin of Cornwell's Sharpe — quieter, funnier, profoundly companionable once you tune in to the rhythm."),
    Book("hhhh", "HHhH", "Laurent Binet", "historical",
         "The 1942 plot to assassinate Reinhard Heydrich, told as a meta-narrative about the difficulty of telling it. Slim, riveting — Robert Harris would adore this one."),

    # Literary speculative
    Book("piranesi", "Piranesi", "Susanna Clarke", "literary",
         "A man lives alone in an infinite labyrinth of statues and tides. Short, strange, perfect — Le Guin would have loved it."),
    Book("never-let-me-go", "Never Let Me Go", "Kazuo Ishiguro", "literary",
         "Quiet dystopia disguised as a boarding-school memoir. Devastating in the same way Cloud Atlas is devastating, but more concentrated."),
    Book("jonathan-strange", "Jonathan Strange & Mr Norrell", "Susanna Clarke", "literary",
         "Two magicians revive English magic during the Napoleonic Wars. Footnoted, funny, immense — the texture of Le Guin with the wit of an Austen novel."),
    Book("memory-police", "The Memory Police", "Yoko Ogawa", "literary",
         "On a small island, things vanish from memory one by one — birds, ribbons, novels, body parts. Ishiguro-quiet, Atwood-political, hauntingly beautiful."),
]
```

---

## Stretch (do these only if everything above works)

- `--migrate-from-artifact` flag that imports state from the old browser artifact's localStorage if I can paste a JSON dump.
- A `?` command that uses Claude to give a deeper analysis of any specific book on the page when I ask for it.
- Simple per-book notes: a free-text comment I can add when marking, persisted alongside the mark, included in future API calls so Claude understands *why* I loved or passed.
- Export the current list and history to a Markdown file with `--export ~/Desktop/books.md`.

---

## What I want from you

Build it all in one go. Test it manually before declaring done — at minimum:

1. Run with `--test-api` and confirm it talks to the API.
2. Run with `--dry-run`, mark a book, generate fresh picks, verify the placeholder replacement works end-to-end.
3. Run for real with one marked book, verify a real Claude reply replaces the slot.
4. Force a parse failure (e.g. by setting an invalid model name) and verify the error path doesn't corrupt state.

Tell me which Python version you tested against, what dependencies pinned at what versions, and how to run it. A short `README.md` is fine.
