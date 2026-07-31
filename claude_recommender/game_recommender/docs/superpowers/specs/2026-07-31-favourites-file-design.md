# Design: `--favourites-file` option for new-user creation

Date: 2026-07-31
Status: approved

## Goal

Allow `python server.py --new-user NAME --favourites-file PATH` to read the
new user's favourite games from a text file instead of typing them line by
line at the interactive prompt. Existing users and all other behaviour are
unaffected.

## Behaviour

- New argparse option `--favourites-file PATH` on `server.py`.
- Only meaningful together with `--new-user`; if given without it, the
  command exits with a clear error.
- During user creation, when the flag is present:
  - Read the file as UTF-8.
  - Strip surrounding whitespace and drop blank lines.
  - Use the result as the favourites text that would otherwise come from the
    interactive multiline prompt.
  - Print a short confirmation, e.g. `Read 12 favourites from games.txt`.
- The display-name and genre-hint prompts remain interactive, unchanged.
- File format: free-form, one favourite per line — identical to what the
  interactive prompt accepts (e.g. `Hollow Knight — Team Cherry`).

## Error handling

- File missing or unreadable → clear error message, exit code 1.
- File empty after stripping blank lines → same "No favourites given —
  cannot build a profile" abort as the interactive path.

## Out of scope

- No changes to profile format, storage layout, or the server/web paths.
- No non-interactive flags for display name or genre hint.

## Testing

- Interactive path without the flag: unchanged.
- `--favourites-file` with a sample file: favourites text is read correctly
  (verified up to the Claude call; the API call itself is unchanged).
- Missing file and empty file: correct error messages and exit codes.
- `--favourites-file` without `--new-user`: rejected.
