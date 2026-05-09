#!/usr/bin/env python3
"""Personal book recommender CLI.

Run:
    python recommender.py            # normal interactive mode
    python recommender.py --dry-run  # no API calls; placeholder picks
    python recommender.py --test-api # one minimal API call, then exit
    python recommender.py -v / -vv   # verbose API logging
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

from data import (
    Book,
    Mark,
    GENRES,
    ORIGINAL_FAVOURITES,
    genre_label,
    genre_sub,
)
from state import State, load_state, save_state, append_log, humanize_age
import claude as cl


console = Console()


# ---------------------------------------------------------------------------
# Sorting / scoring
# ---------------------------------------------------------------------------

def book_sort_rank(book: Book, mark: Mark) -> int:
    """Lower rank = higher in the list. Order: loved → unmarked → read-only → passed."""
    if mark.like == "love":
        return 0
    if mark.like == "meh":
        return 3
    if mark.read:
        return 2
    return 1


def genre_score(genre_key: str, state: State) -> float:
    score = 0.0
    for b in state.current_books:
        if b.genre != genre_key:
            continue
        m = state.get_mark(b.id)
        if m.like == "love":
            score += 2.0
        elif m.like == "meh":
            score -= 2.0
        elif m.read and m.like is None:
            score += 0.25
    return score


def ordered_genres(state: State) -> list[dict]:
    """Genres sorted by score desc, original order as tiebreak."""
    indexed = list(enumerate(GENRES))
    indexed.sort(key=lambda pair: (-genre_score(pair[1]["key"], state), pair[0]))
    return [g for _, g in indexed]


def books_in_genre(state: State, genre_key: str) -> list[Book]:
    books = [b for b in state.current_books if b.genre == genre_key]
    books.sort(key=lambda b: book_sort_rank(b, state.get_mark(b.id)))
    return books


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def clear_and_header(title: str, sub: Optional[str] = None) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    if sub:
        console.print(f"[italic dim]{sub}[/italic dim]")
    console.print()


def render_top_screen(state: State) -> None:
    marked = state.marked_books()
    loved_count = len(state.loved_books())
    passed_count = len(state.passed_books())

    clear_and_header("Your book recommender")
    console.print(
        f"  [bold]Marked:[/bold] {len(marked)}    "
        f"[green]Loved:[/green] {loved_count}    "
        f"[yellow]Passed:[/yellow] {passed_count}    "
        f"[dim]In history:[/dim] {len(state.history)}"
    )
    console.print()
    console.print("  [bold]Genres[/bold]")
    for i, g in enumerate(ordered_genres(state), start=1):
        gbooks = [b for b in state.current_books if b.genre == g["key"]]
        marked_in_g = sum(1 for b in gbooks if state.get_mark(b.id).is_marked())
        suffix = f" · {marked_in_g} marked" if marked_in_g else ""
        console.print(f"    {i}. {g['label']:<28} {len(gbooks)} books{suffix}")
    console.print()
    g_hint = "(nothing marked yet)" if not marked else f"(replace {len(marked)} marked book{'s' if len(marked) != 1 else ''})"
    console.print("  [bold]Commands[/bold]")
    console.print("    [cyan]\\[number][/cyan]   open a genre")
    console.print(f"    [cyan]g[/cyan]          generate fresh picks {g_hint}")
    console.print("    [cyan]h[/cyan]          show marked-books history")
    console.print("    [cyan]s[/cyan]          show stats")
    console.print("    [cyan]r[/cyan]          reset all marks (history kept)")
    console.print("    [cyan]q[/cyan]          quit")
    console.print()


def render_genre_screen(state: State, genre_key: str) -> list[Book]:
    label = genre_label(genre_key)
    sub = genre_sub(genre_key)
    clear_and_header(label, sub)
    books = books_in_genre(state, genre_key)
    for i, b in enumerate(books, start=1):
        m = state.get_mark(b.id)
        tag = ""
        title_line_style = ""
        if m.like == "love":
            tag = "[bold green](LOVED)[/bold green] "
        elif m.like == "meh":
            tag = "[yellow](PASSED)[/yellow] "
            title_line_style = "dim"
        elif m.read:
            tag = "[dim](READ)[/dim] "
            title_line_style = "dim"
        fresh = " [magenta]✦ fresh[/magenta]" if b.is_fresh else ""
        title_text = f"{tag}{b.title} — {b.author}{fresh}"
        if title_line_style:
            console.print(f"  [{title_line_style}][{i}][/{title_line_style}]  [{title_line_style}]{title_text}[/{title_line_style}]")
            console.print(f"       [{title_line_style}]{b.why}[/{title_line_style}]")
        else:
            console.print(f"  [cyan][{i}][/cyan]  {title_text}")
            console.print(f"       [italic]{b.why}[/italic]")
        console.print()
    console.print("  [bold]Commands[/bold]")
    console.print("    [cyan]\\[number][/cyan]   mark a book")
    console.print("    [cyan]b[/cyan]          back")
    console.print("    [cyan]g[/cyan]          generate fresh picks")
    console.print()
    return books


def render_mark_screen(book: Book, current_mark: Mark) -> None:
    clear_and_header(f'"{book.title}" by {book.author}')
    if current_mark.is_marked():
        if current_mark.like == "love":
            console.print("  [dim]current:[/dim] [green]LOVED[/green]")
        elif current_mark.like == "meh":
            console.print("  [dim]current:[/dim] [yellow]PASSED[/yellow]")
        elif current_mark.read:
            console.print("  [dim]current:[/dim] [dim]READ[/dim]")
        console.print()
    console.print("  [cyan]l[/cyan]   loved it")
    console.print("  [cyan]p[/cyan]   not for me")
    console.print("  [cyan]r[/cyan]   mark read (no strong opinion)")
    console.print("  [cyan]u[/cyan]   unmark")
    console.print("  [cyan]c[/cyan]   cancel")
    console.print()


def render_history_screen(state: State) -> None:
    clear_and_header(f"Marked books history ({len(state.history)})")
    if not state.history:
        console.print("  [dim]No history yet. Mark a book and run 'g' to generate replacements.[/dim]")
        console.print()
    for h in state.history:
        if h.mark == "loved":
            tag = "[bold green]LOVED[/bold green] "
        elif h.mark == "passed":
            tag = "[yellow]PASSED[/yellow]"
        else:
            tag = "[dim]READ[/dim]  "
        console.print(f"  {tag}  {h.title} — {h.author}")
        console.print(f"          [dim]{genre_label(h.genre).lower()} · replaced {humanize_age(h.replaced_at)}[/dim]")
        console.print()
    console.print("  [bold]Commands:[/bold] [cyan]b[/cyan] back · [cyan]c[/cyan] clear history · [cyan]q[/cyan] quit")
    console.print()


def render_stats(state: State) -> None:
    clear_and_header("Stats")
    loved = len(state.loved_books())
    passed = len(state.passed_books())
    read_only = sum(
        1 for b in state.current_books
        if (m := state.get_mark(b.id)) and m.read and m.like is None
    )
    fresh = sum(1 for b in state.current_books if b.is_fresh)
    console.print(f"  Books currently listed:   {len(state.current_books)}")
    console.print(f"  Of those, fresh picks:    {fresh}")
    console.print(f"  [green]Loved (current list):[/green]    {loved}")
    console.print(f"  [yellow]Passed (current list):[/yellow]   {passed}")
    console.print(f"  [dim]Read-only (current):[/dim]     {read_only}")
    console.print(f"  History entries:          {len(state.history)}")
    console.print()
    console.print("  [bold]Genre scores[/bold] (drives genre ordering)")
    for g in ordered_genres(state):
        console.print(f"    {g['label']:<32} {genre_score(g['key'], state):+.2f}")
    console.print()
    console.print("  Press [cyan]b[/cyan] to go back.")
    console.print()


# ---------------------------------------------------------------------------
# Generate fresh picks
# ---------------------------------------------------------------------------

def make_client():
    """Create an Anthropic client; raises if API key missing."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[bold red]ANTHROPIC_API_KEY is not set.[/bold red]\n"
            "Set it with:  [cyan]export ANTHROPIC_API_KEY=...[/cyan]"
        )
        sys.exit(2)
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def generate_fresh_picks(
    state: State,
    dry_run: bool,
    verbose: int,
) -> None:
    marked = state.marked_books()
    if not marked:
        console.print("[dim]Nothing is marked. Mark some books first.[/dim]")
        console.print()
        return

    # Build batches. Each entry is (Book, mark_label).
    batched: list[list[tuple[Book, str]]] = []
    current_batch: list[tuple[Book, str]] = []
    for b in marked:
        m = state.get_mark(b.id)
        label = m.label() or "read"
        current_batch.append((b, label))
        if len(current_batch) == cl.BATCH_SIZE:
            batched.append(current_batch)
            current_batch = []
    if current_batch:
        batched.append(current_batch)

    n_batches = len(batched)
    total = len(marked)
    if dry_run:
        console.print(f"[dim]--dry-run:[/dim] {total} marked. Generating placeholders…")
    else:
        console.print(f"{total} books are marked. Asking Claude for tailored replacements…")
    console.print()

    client = None
    if not dry_run:
        client = make_client()

    n_replaced = 0
    n_failed = 0
    for bi, batch in enumerate(batched, start=1):
        size = len(batch)
        console.print(f"  [bold]Batch {bi} of {n_batches}[/bold] ({size} replacement{'s' if size != 1 else ''})…", end=" ")

        if dry_run:
            time.sleep(0.1)
            picks = [cl.make_placeholder_pick(b) for (b, _) in batch]
            latency = 0.1
        else:
            user_payload = cl.build_user_payload(state, ORIGINAL_FAVOURITES, batch)
            if verbose >= 1:
                console.print()
                console.print(f"    [dim]model:[/dim] {cl.MODEL}  [dim]prompt_chars:[/dim] {len(user_payload)}")
            try:
                raw, latency = cl.call_claude(client, user_payload, verbose=verbose)
            except cl.APIError as e:
                console.print()
                console.print(f"    [red]✗ batch failed[/red] ({e.kind}): {e}")
                n_failed += size
                if e.kind in ("auth", "rate_limit"):
                    if e.kind == "rate_limit" and e.retry_after:
                        console.print(f"    [yellow]Wait {e.retry_after:.0f}s and try again.[/yellow]")
                    elif e.kind == "auth":
                        console.print("    [yellow]Check your ANTHROPIC_API_KEY.[/yellow]")
                    console.print(f"    [dim]Stopping — {n_failed} failed, {n_replaced} replaced so far.[/dim]")
                    break
                # parse / network / timeout / other → skip batch and continue
                continue

            if verbose >= 1:
                preview = raw if verbose >= 2 else raw[:1000] + ("…[truncated]" if len(raw) > 1000 else "")
                console.print(f"    [dim]response_chars:[/dim] {len(raw)}  [dim]latency:[/dim] {latency:.2f}s")
                console.print(Panel(preview or "[empty]", title="raw response", border_style="dim"))

            try:
                raw_picks = cl.parse_response(raw)
            except cl.APIError as e:
                console.print()
                console.print(f"    [red]✗ couldn't parse this — leaving state unchanged[/red]")
                console.print(f"    [dim]{e}[/dim]")
                console.print(Panel(raw[:500] + ("…" if len(raw) > 500 else ""), title="first 500 chars", border_style="red"))
                n_failed += size
                continue

            picks: list[dict] = []
            valid_pairs: list[tuple[Book, dict]] = []
            for (old, _label), pick_raw in zip(batch, raw_picks[:size]):
                cleaned = cl.validate_pick(pick_raw, old.genre)
                if cleaned is None:
                    console.print(f"    [yellow]skipped malformed pick for '{old.title}'[/yellow]")
                    continue
                valid_pairs.append((old, cleaned))
                picks.append(cleaned)

            if not valid_pairs:
                console.print()
                console.print("    [red]✗ no valid picks in this batch[/red]")
                n_failed += size
                continue

            # Adjust failure tally for any malformed ones
            n_failed += size - len(valid_pairs)
            console.print(f"done in {latency:.1f}s")

        # Commit successful pairs
        if dry_run:
            results = cl.commit_replacements(state, list(zip([b for b, _ in batch], picks)))
        else:
            results = cl.commit_replacements(state, valid_pairs)
        save_state(state)
        n_replaced += len(results)

        for old, new, label in results:
            console.print()
            console.print(f"  [green]✓[/green] [bold]{genre_label(new.genre)}[/bold]")
            console.print(f"    [dim]REPLACED:[/dim] {old.title}  ([italic]{label}[/italic])")
            console.print(f"    [dim]WITH:[/dim]     {new.title} — {new.author}")
            console.print(f'              [italic]"{new.why}"[/italic]')
        console.print()

    console.print(f"[bold]{n_replaced} replaced, {n_failed} failed.[/bold] Returning to main menu.")
    console.print()


# ---------------------------------------------------------------------------
# Mark flow
# ---------------------------------------------------------------------------

def prompt_mark_book(state: State, book: Book) -> None:
    while True:
        render_mark_screen(book, state.get_mark(book.id))
        choice = input("> ").strip().lower()
        if choice == "l":
            state.set_mark(book.id, Mark(read=True, like="love"))
            save_state(state)
            return
        if choice == "p":
            state.set_mark(book.id, Mark(read=True, like="meh"))
            save_state(state)
            return
        if choice == "r":
            state.set_mark(book.id, Mark(read=True, like=None))
            save_state(state)
            return
        if choice == "u":
            state.set_mark(book.id, Mark(read=False, like=None))
            save_state(state)
            return
        if choice == "c":
            return
        console.print("[red]Unknown option.[/red]")


def genre_loop(state: State, genre_key: str, dry_run: bool, verbose: int) -> None:
    while True:
        books = render_genre_screen(state, genre_key)
        choice = input("> ").strip().lower()
        if choice == "b":
            return
        if choice == "g":
            generate_fresh_picks(state, dry_run=dry_run, verbose=verbose)
            input("Press enter to continue…")
            return  # go back to top so user sees re-ranked everything
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(books):
                prompt_mark_book(state, books[idx - 1])
                continue
            console.print("[red]No book at that number.[/red]")
            continue
        console.print("[red]Unknown command.[/red]")


def history_loop(state: State) -> None:
    while True:
        render_history_screen(state)
        choice = input("> ").strip().lower()
        if choice == "b":
            return
        if choice == "q":
            sys.exit(0)
        if choice == "c":
            confirm = input("Clear ALL history? This can't be undone. (y/N) ").strip().lower()
            if confirm == "y":
                state.history.clear()
                save_state(state)
                console.print("[dim]History cleared.[/dim]")
            continue
        console.print("[red]Unknown command.[/red]")


def reset_marks(state: State) -> None:
    confirm = input("Reset all marks? (history kept) (y/N) ").strip().lower()
    if confirm == "y":
        state.marks.clear()
        save_state(state)
        console.print("[dim]All marks cleared.[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def top_loop(state: State, dry_run: bool, verbose: int) -> None:
    while True:
        render_top_screen(state)
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            console.print()
            return
        if choice == "q":
            return
        if choice == "g":
            generate_fresh_picks(state, dry_run=dry_run, verbose=verbose)
            input("Press enter to continue…")
            continue
        if choice == "h":
            history_loop(state)
            continue
        if choice == "s":
            render_stats(state)
            input("> ")
            continue
        if choice == "r":
            reset_marks(state)
            continue
        if choice.isdigit():
            idx = int(choice)
            ordered = ordered_genres(state)
            if 1 <= idx <= len(ordered):
                genre_loop(state, ordered[idx - 1]["key"], dry_run=dry_run, verbose=verbose)
                continue
            console.print("[red]No genre at that number.[/red]")
            continue
        console.print("[red]Unknown command.[/red]")


def run_test_api() -> int:
    """Make a minimal API call and exit. Returns shell exit code."""
    client = make_client()
    import anthropic

    console.print(f"Testing API with model [cyan]{cl.MODEL}[/cyan]…")
    start = time.time()
    try:
        resp = client.messages.create(
            model=cl.MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": "say hi"}],
            timeout=30.0,
        )
        latency = time.time() - start
    except anthropic.AuthenticationError as e:
        console.print(f"[red]Auth error:[/red] {e}")
        return 2
    except anthropic.APIError as e:
        console.print(f"[red]API error:[/red] {type(e).__name__}: {e}")
        return 3

    text = ""
    if resp.content:
        text = getattr(resp.content[0], "text", "") or ""
    console.print(f"  model:    {resp.model}")
    console.print(f"  latency:  {latency:.2f}s")
    console.print(f"  response: {text!r}")
    append_log({
        "model": cl.MODEL, "prompt_chars": 6, "response_chars": len(text),
        "latency_ms": int(latency * 1000), "test_api": True,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Personal book recommender CLI")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v: show API metadata and first 1000 chars; -vv: full raw response.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't call the API; generate placeholder replacements.")
    parser.add_argument("--test-api", action="store_true",
                        help="Make one minimal API call and exit.")
    args = parser.parse_args()

    if args.test_api:
        return run_test_api()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[bold red]ANTHROPIC_API_KEY is not set.[/bold red]\n"
            "Set it with:  [cyan]export ANTHROPIC_API_KEY=...[/cyan]\n"
            "Or run with [cyan]--dry-run[/cyan] to test the UI without an API key."
        )
        return 2

    state = load_state()
    try:
        top_loop(state, dry_run=args.dry_run, verbose=args.verbose)
    except KeyboardInterrupt:
        console.print()
        console.print("[dim]Interrupted. State saved.[/dim]")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
