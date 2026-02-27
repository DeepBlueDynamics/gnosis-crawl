"""Console + JSON report generator for combat results.

Usage:
    python -m combat.report                # read from combat/results.json
    python -m combat.report path/to.json   # read from custom path
"""
from __future__ import annotations

import json
import pathlib
import sys

from tabulate import tabulate

RESULTS_PATH = pathlib.Path(__file__).parent / "results.json"


def _load(path: pathlib.Path | None = None) -> dict:
    p = path or RESULTS_PATH
    if not p.exists():
        print(f"No results file at {p}. Run the combat suite first:")
        print("  pytest combat/ -m combat -v")
        sys.exit(1)
    return json.loads(p.read_text())


def _winner(row: dict, metric: str, lower_is_better: bool = True) -> str:
    """Return the adapter name with the best value for *metric*."""
    best_name, best_val = None, None
    for name, data in row.items():
        val = data.get(metric)
        if val is None:
            continue
        # Skip failed adapters for speed/batch, but allow 0 values for quality
        if not data.get("success", True) and metric in ("elapsed_ms", "total_ms"):
            continue
        if best_val is None or (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val = val
            best_name = name
    return best_name or "-"


def _adapters_from(data: dict) -> list[str]:
    """Collect all adapter names seen in any section."""
    names: set[str] = set()
    for section in data.values():
        for row in section.values():
            names.update(row.keys())
    # Stable order: Grub first, then alphabetical
    ordered = sorted(names - {"Grub"})
    if "Grub" in names:
        ordered.insert(0, "Grub")
    return ordered


def _fmt(val, fmt=".0f"):
    """Format a numeric value or return dash for missing."""
    if val is None or val == "-":
        return "-"
    try:
        return f"{val:{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def print_report(data: dict) -> None:
    adapters = _adapters_from(data)

    if not adapters:
        print("No adapter data found in results.")
        return

    print()
    print("=" * 72)
    print("                    GRUB COMBAT REPORT")
    print("=" * 72)
    print(f"  Adapters: {', '.join(adapters)}")

    wins = {a: 0 for a in adapters}
    total_races = 0

    # --- Speed ---
    if data.get("speed"):
        print()
        print("SPEED  (single URL, ms -lower is better)")
        print("-" * 72)
        headers = ["URL"] + adapters + ["Winner"]
        rows = []
        for url_key, row in data["speed"].items():
            short = url_key[:35]
            cells = [_fmt(row.get(a, {}).get("elapsed_ms")) for a in adapters]
            winner = _winner(row, "elapsed_ms", lower_is_better=True)
            if winner in wins:
                wins[winner] += 1
            total_races += 1
            rows.append([short] + cells + [winner])
        print(tabulate(rows, headers=headers, tablefmt="grid", numalign="right"))

        # Phase breakdown for Grub
        grub_phases = []
        for url_key, row in data["speed"].items():
            t = row.get("Grub", {}).get("timings", {})
            if t:
                grub_phases.append([
                    url_key[:35],
                    _fmt(t.get("navigation_ms")),
                    _fmt(t.get("content_ms")),
                    _fmt(t.get("visible_text_ms")),
                    _fmt(t.get("markdown_ms")),
                    _fmt(t.get("total_ms")),
                ])
        if grub_phases:
            print()
            print("GRUB PHASE BREAKDOWN  (server-side ms)")
            print("-" * 72)
            print(tabulate(
                grub_phases,
                headers=["URL", "nav", "content", "visible_text", "markdown", "server_total"],
                tablefmt="grid",
                numalign="right",
            ))

    # --- Quality: word count ---
    if data.get("quality"):
        print()
        print("QUALITY  (word count -higher is better)")
        print("-" * 72)
        headers = ["URL"] + adapters + ["Winner"]
        rows = []
        for url_key, row in data["quality"].items():
            short = url_key[:35]
            cells = [_fmt(row.get(a, {}).get("word_count")) for a in adapters]
            winner = _winner(row, "word_count", lower_is_better=False)
            rows.append([short] + cells + [winner])
        print(tabulate(rows, headers=headers, tablefmt="grid", numalign="right"))

        # --- Quality: content ratio ---
        print()
        print("QUALITY  (content ratio = markdown/html -higher is better)")
        print("-" * 72)
        headers = ["URL"] + adapters + ["Winner"]
        rows = []
        for url_key, row in data["quality"].items():
            short = url_key[:35]
            cells = [_fmt(row.get(a, {}).get("content_ratio"), ".3f") for a in adapters]
            winner = _winner(row, "content_ratio", lower_is_better=False)
            rows.append([short] + cells + [winner])
        print(tabulate(rows, headers=headers, tablefmt="grid", numalign="right"))

    # --- Batch ---
    if data.get("batch"):
        print()
        print("BATCH THROUGHPUT  (lower ms is better)")
        print("-" * 72)
        headers = ["Batch"] + [f"{a} (ms)" for a in adapters] + [f"{a} (%ok)" for a in adapters] + ["Winner"]
        rows = []
        for size_key, row in data["batch"].items():
            ms_cells = [_fmt(row.get(a, {}).get("total_ms")) for a in adapters]
            ok_cells = [_fmt(row.get(a, {}).get("success_rate"), ".0%") if row.get(a, {}).get("success_rate") is not None else "-" for a in adapters]
            winner = _winner(row, "total_ms", lower_is_better=True)
            if winner in wins:
                wins[winner] += 1
            total_races += 1
            rows.append([size_key] + ms_cells + ok_cells + [winner])
        print(tabulate(rows, headers=headers, tablefmt="grid", numalign="right"))

        # Per-URL throughput
        print()
        print("BATCH PER-URL  (ms/url -lower is better)")
        print("-" * 72)
        headers = ["Batch"] + adapters
        rows = []
        for size_key, row in data["batch"].items():
            cells = [_fmt(row.get(a, {}).get("per_url_ms")) for a in adapters]
            rows.append([size_key] + cells)
        print(tabulate(rows, headers=headers, tablefmt="grid", numalign="right"))

    # --- Scorecard ---
    print()
    print("=" * 72)
    print("                       SCORECARD")
    print("=" * 72)

    score_rows = []

    # Speed wins
    if data.get("speed"):
        n = len(data["speed"])
        cells = []
        for a in adapters:
            count = sum(1 for row in data["speed"].values() if _winner(row, "elapsed_ms", True) == a)
            cells.append(f"{count}/{n}")
        score_rows.append(["Speed (single URL)"] + cells)

    # Batch wins
    if data.get("batch"):
        n = len(data["batch"])
        cells = []
        for a in adapters:
            count = sum(1 for row in data["batch"].values() if _winner(row, "total_ms", True) == a)
            cells.append(f"{count}/{n}")
        score_rows.append(["Batch throughput"] + cells)

    # Quality word count wins
    if data.get("quality"):
        n = len(data["quality"])
        cells = []
        for a in adapters:
            count = sum(1 for row in data["quality"].values() if _winner(row, "word_count", False) == a)
            cells.append(f"{count}/{n}")
        score_rows.append(["Quality (words)"] + cells)

        cells = []
        for a in adapters:
            count = sum(1 for row in data["quality"].values() if _winner(row, "content_ratio", False) == a)
            cells.append(f"{count}/{n}")
        score_rows.append(["Quality (ratio)"] + cells)

    if score_rows:
        print(tabulate(score_rows, headers=["Category"] + adapters, tablefmt="grid", stralign="center"))

    print()
    print("=" * 72)


def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    data = _load(path)
    print_report(data)


if __name__ == "__main__":
    main()
