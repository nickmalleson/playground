# Personal book recommender (CLI)

Terminal-based personal book recommender that talks to the Anthropic API directly. Built from `book-recommender-build-spec.md`.

## What it does

- Shows 24 curated books across 6 genres.
- Lets you mark each book as **loved**, **not for me (passed)**, or **read** (no strong opinion).
- Persists marks across runs in `~/.book_recommender/state.json`.
- Sorts within each genre: loved → unmarked → read → passed.
- Re-ranks the genre sections themselves so genres you've loved more rise to the top.
- On `g`, asks Claude (`claude-sonnet-4-6`) for one tailored replacement per marked book — same genre, tailored to whether you loved/passed/read it.
- Keeps a permanent history of every replaced book.

## Setup

Requires Python 3.10+. Built and tested against:

- Python 3.11.15
- `anthropic==0.99.0`
- `rich==15.0.0`

```bash
conda create -n book-recommender python=3.11 -y
conda activate book-recommender
pip install anthropic rich

export ANTHROPIC_API_KEY=sk-ant-...
```

## Run it

```bash
python recommender.py
```

### Flags

| Flag | What it does |
| --- | --- |
| `--dry-run` | Skip the API entirely; generate placeholder replacements. Lets you exercise marking, state, history, and re-ranking with no key needed. |
| `--test-api` | Make one minimal API call (`max_tokens=20`) and exit. Prints model, latency, and response. Use to verify the key and connectivity. |
| `-v` / `--verbose` | Log API metadata (model, prompt size, response size, latency) and the first 1000 chars of each raw response. |
| `-vv` | Same, but with full untruncated raw response. |

## Files on disk

| Path | Purpose |
| --- | --- |
| `~/.book_recommender/state.json` | Current book list, marks, and history. Atomic writes. |
| `~/.book_recommender/log.jsonl` | Append-only API call log: `{ts, model, prompt_chars, response_chars, latency_ms, error?}`. With `-v`, also includes the prompt and raw response. |
| `~/.book_recommender/state.json.corrupt-{ts}` | Auto-created if the state file fails to parse — your previous file is preserved here while the app starts fresh. |

## Project layout

| File | Role |
| --- | --- |
| `data.py` | `Book`, `Mark`, `HistoryEntry` dataclasses; `GENRES`, `ORIGINAL_FAVOURITES`, `SEED_BOOKS`. |
| `state.py` | Load/save state with atomic writes; corrupt-file recovery; logging; relative-time helper. |
| `claude.py` | Anthropic API client, system prompt, batching, forgiving response parser, replacement commit step. |
| `recommender.py` | CLI loop, screens (top, genre, mark, history, stats), rich rendering. |

## How replacements work

When you press `g`:

1. All marked books are gathered. If there are more than 6, they're split into batches of 6.
2. Each batch is sent to Claude with the originals favourites, an avoid list (current books + history), the loved/passed-so-far context, and a slot list (genre + replacing-title + mark).
3. The response is parsed forgivingly: code fences are stripped, dict-wrapped arrays are unwrapped, and a balanced-bracket scan recovers arrays buried in prose.
4. For each `(marked_book, pick)` pair: a `HistoryEntry` is appended, a fresh `Book` is built (genre forced to match the slot), the slot in `current_books` is replaced, and the mark is dropped. Genre is forced — Claude can't change a slot's genre.
5. **Each batch is committed to disk before the next batch starts.** A crash or rate-limit halfway through doesn't lose anything.

## Error handling

- **Missing `ANTHROPIC_API_KEY`** → exits with the export instruction (unless `--dry-run`).
- **Auth or rate-limit errors** → batch is skipped, processing stops, state untouched. Rate limits print the retry-after if the API provided one.
- **Network / timeout / parse errors** → batch is skipped, state untouched, processing continues with the next batch.
- **Parse failure** → first 500 chars of the raw response are shown so you can see what Claude actually said.
- **Corrupt state file on load** → backed up to `state.json.corrupt-{ts}`, fresh state created.
- **Crashes** → traceback printed to stderr; nothing is swallowed.

## Tested scenarios

This was tested manually against the four scenarios from the spec:

1. **`--test-api`** — surfaced the missing-key error path correctly. Live call was not run in the build environment because no key was set; run it yourself to verify connectivity.
2. **`--dry-run` end-to-end** — marked books across multiple genres, verified within-genre and across-genre re-ordering, viewed stats and history, generated placeholder replacements, confirmed history grew and marks cleared.
3. **Real call with one marked book** — *not exercised in the build environment* (no API key available). The dry-run path uses the same `commit_replacements` step as the live path, so once the API call returns and parses cleanly the rest of the flow is identical.
4. **Forced parse failure** — simulated by monkey-patching `call_claude` to return non-JSON. Verified the failure path leaves `current_books`, `marks`, and `history` untouched and shows a clear error with the first 500 chars of the response.

Also exercised:

- Corrupt `state.json` recovery (file backed up, fresh state loaded).
- Auth-error stop path (processing halts, state untouched).
