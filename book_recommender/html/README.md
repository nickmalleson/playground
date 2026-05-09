# Book recommender — local web app

A small Python web server (Flask) plus a single-page HTML frontend.
The browser keeps the nice UI; all Claude API calls go through Python so you can see every request and response in your terminal.

## One-time setup

### Option A — conda (recommended)

Create and activate the environment:

```
conda env create -f environment.yml
conda activate book_recommender
```

### Option B — pip

```
pip install -r requirements.txt
```

Set your Anthropic API key (do this in every shell where you run the server, or add to your `~/.zshrc` / `~/.bashrc`):

```
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run it

```
python server.py
```

You should see startup output like:

```
══════════════════════════════════════════════════════════
Book recommender server
  working dir : /Users/.../book_recommender/html
  state file  : /Users/.../book_recommender/html/state.json
  log file    : /Users/.../book_recommender/html/server.log
  model       : claude-sonnet-4-6
  batch size  : 6
  timeout     : 90.0s per call
  api key     : sk-ant-abc…wxyz
  listening on: http://localhost:5050
  open this URL in your browser to use the app:
     http://localhost:5050
══════════════════════════════════════════════════════════
```

Open `http://localhost:5050` in your browser.

## Command-line flags

```
python server.py --debug             # verbose logging (full prompts, full responses)
python server.py --port 5070         # different port
```

## What you'll see in the terminal

Every browser action shows up in your terminal in real time:

- `mark id='snow-crash' patch={'like': 'love'}` — every time you mark a book
- `→ Claude  model=…  prompt_chars=2453  slots=3` — every API call out
- `← Claude  latency=8.4s  resp_chars=412  in_tok=… out_tok=…` — every reply
- Full traceback on any error

With `--debug`, you also get the complete prompt and response printed.

## Files

- **`server.py`** — Flask backend with all the Claude logic
- **`index.html`** — single-page frontend
- **`state.json`** — your marks, history, current book list (created on first run; safe to back up, delete, or hand-edit)
- **`server.log`** — append-only log of everything the server does
- **`environment.yml`** — conda environment (Python 3.14, Flask, Anthropic)
- **`requirements.txt`** — pip dependencies

## Useful endpoints (for poking around)

- `GET  /api/state` — full current state
- `POST /api/mark`  — `{"id": "...", "patch": {"like": "love"}}`
- `POST /api/clear-marks`
- `POST /api/clear-history`
- `POST /api/fresh-picks` — generate replacements for marked books
- `GET  /api/test` — minimal Claude ping for connectivity check (also surfaced as the "Test Claude" button in the UI)

## Troubleshooting

**"Address already in use"** — port 5050 is taken. Run `python server.py --port 5070` (or any free port).

**The page says "server unreachable"** — Python server isn't running, or it's on a different port. Check the terminal where you started it.

**Fresh picks fail** — look at the server terminal. The error and a snippet of Claude's actual response will be there. The browser also shows the same info in the red error bar at the top of the page.

**Reset everything** — close the server, delete `state.json`, restart. Or use the in-page "Reset marks" / "Clear history" buttons.
