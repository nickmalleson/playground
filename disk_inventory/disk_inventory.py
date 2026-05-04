#!/usr/bin/env python3
"""Disk usage treemap — like Disk Inventory X, in Python.

Scans a directory tree and writes an interactive HTML treemap (via Plotly)
showing where disk space is used. Click a tile to drill in; hover for the
full path and size. Folders are drawn as boxes containing their children,
so clusters of small files in one directory are easy to spot.

Usage:
    python disk_inventory.py                # scan from /
    python disk_inventory.py ~/Downloads    # scan a specific folder
    python disk_inventory.py / -o out.html  # custom output path
"""

import argparse
import os
import stat
import sys
import webbrowser
from collections import defaultdict

import plotly.graph_objects as go


# Special macOS directories we never descend into. These are either virtual
# filesystems, system-managed, or other mounted volumes — none of which
# the user can usefully delete from.
SKIP_DIR_NAMES = {
    ".Spotlight-V100", ".fseventsd", ".DocumentRevisions-V100",
    ".TemporaryItems", ".Trashes", ".vol",
    ".PKInstallSandboxManager", ".PKInstallSandboxManager-SystemSoftware",
}
SKIP_ABS_PATHS = {
    "/System", "/private", "/dev", "/cores", "/Network", "/Volumes",
}


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def scan(root):
    """Walk root, returning a list of (path, size) for every regular file."""
    files = []
    n = 0
    for dirpath, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False, onerror=lambda e: None
    ):
        # Filter dirnames in-place so os.walk doesn't descend into skipped dirs.
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIR_NAMES
            and os.path.join(dirpath, d) not in SKIP_ABS_PATHS
        ]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                st = os.lstat(fpath)
            except OSError:
                continue
            # Only count regular files (skip symlinks, devices, sockets).
            if not stat.S_ISREG(st.st_mode):
                continue
            files.append((fpath, st.st_size))
            n += 1
            if n % 50000 == 0:
                total = sum(s for _, s in files)
                print(f"  scanned {n:,} files ({human(total)})…",
                      file=sys.stderr)
    return files


def build_treemap_data(files, root, min_fraction):
    """Build node arrays for a Plotly treemap.

    Folder totals always reflect everything inside them. Individual files
    (and whole subtrees) smaller than `min_fraction * total_size` are
    rolled up into a single '(other small items)' tile inside their parent
    so the treemap stays readable, while preserving the parent's true area.
    """
    total_size = sum(s for _, s in files)
    threshold = total_size * min_fraction

    dir_total = defaultdict(int)        # bytes under each directory
    dir_file_count = defaultdict(int)   # recursive file count
    dir_files = defaultdict(list)       # dir -> [(name, path, size)]
    dir_subdirs = defaultdict(set)      # dir -> {immediate subdir paths}

    for fpath, size in files:
        parent = os.path.dirname(fpath)
        dir_files[parent].append((os.path.basename(fpath), fpath, size))

        # Walk up to root, accumulating totals and registering subdirs.
        cur = parent
        while True:
            dir_total[cur] += size
            dir_file_count[cur] += 1
            if cur == root:
                break
            par = os.path.dirname(cur)
            if par == cur:  # hit filesystem root before our scan root
                break
            dir_subdirs[par].add(cur)
            cur = par

    ids, labels, parents, values, hover = [], [], [], [], []

    # Root node.
    ids.append(root)
    labels.append(root)
    parents.append("")
    values.append(dir_total[root] or 1)  # avoid 0 which Plotly hides
    hover.append(
        f"<b>{root}</b><br>{human(dir_total[root])} • "
        f"{dir_file_count[root]:,} files"
    )

    # BFS down the directory tree, adding children of each visited dir.
    queue = [root]
    while queue:
        d = queue.pop()
        small_size = 0
        small_count = 0

        for name, fpath, size in dir_files.get(d, []):
            if size >= threshold:
                ids.append(fpath)
                labels.append(name)
                parents.append(d)
                values.append(size)
                hover.append(f"<b>{fpath}</b><br>{human(size)}")
            else:
                small_size += size
                small_count += 1

        for sub in dir_subdirs.get(d, ()):
            sub_size = dir_total[sub]
            if sub_size >= threshold:
                ids.append(sub)
                labels.append(os.path.basename(sub) or sub)
                parents.append(d)
                values.append(sub_size)
                hover.append(
                    f"<b>{sub}/</b><br>{human(sub_size)} • "
                    f"{dir_file_count[sub]:,} files"
                )
                queue.append(sub)
            else:
                small_size += sub_size
                small_count += dir_file_count[sub]

        if small_size > 0:
            ids.append(d + "::(other)")
            labels.append(f"({small_count:,} small items)")
            parents.append(d)
            values.append(small_size)
            hover.append(
                f"<b>{small_count:,} small items in {d}</b><br>"
                f"{human(small_size)} total"
            )

    return ids, labels, parents, values, hover, total_size


def main():
    p = argparse.ArgumentParser(
        description="Interactive disk-usage treemap (HTML output).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("root", nargs="?", default="/",
                   help="root folder to scan")
    p.add_argument("-o", "--output", default="disk_usage.html",
                   help="output HTML file")
    p.add_argument("--min-fraction", type=float, default=0.001,
                   help="bundle items smaller than this fraction of total "
                        "(0.001 = 0.1%%)")
    p.add_argument("--no-open", action="store_true",
                   help="don't auto-open the HTML in a browser")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"error: not a directory: {root}")

    print(f"Scanning {root}…", file=sys.stderr)
    files = scan(root)
    if not files:
        sys.exit("No files found (or all were skipped).")
    total = sum(s for _, s in files)
    print(f"Done: {len(files):,} files, {human(total)} total.",
          file=sys.stderr)

    print("Building treemap…", file=sys.stderr)
    ids, labels, parents, values, hover, total = build_treemap_data(
        files, root, min_fraction=args.min_fraction
    )
    print(f"Treemap has {len(ids):,} tiles.", file=sys.stderr)

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        text=hover,
        branchvalues="total",
        hovertemplate="%{text}<extra></extra>",
        maxdepth=4,  # show 4 levels at once; click to drill deeper
        tiling=dict(packing="squarify"),
        marker=dict(cornerradius=2),
    ))
    fig.update_layout(
        title=f"Disk usage — {root} ({human(total)})",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    out = os.path.abspath(args.output)
    # include_plotlyjs="inline" makes the HTML self-contained (works offline).
    fig.write_html(out, include_plotlyjs="inline")
    print(f"Wrote {out}", file=sys.stderr)

    if not args.no_open:
        webbrowser.open(f"file://{out}")


if __name__ == "__main__":
    main()
