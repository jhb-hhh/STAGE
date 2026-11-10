"""Small console table helpers for artifact outputs."""

import csv
from pathlib import Path


def _shorten(value, width):
    value = str(value)
    if len(value) <= width:
        return value
    return value[: max(0, width - 3)] + "..."


def print_csv_table(csv_path, title=None, max_rows=50, max_col_width=22):
    """Print a compact aligned preview of a CSV file."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"[MISSING] table: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)

    if not rows:
        print(f"[EMPTY] table: {csv_path}")
        return

    header = rows[0]
    body = rows[1:]
    display_body = body[:max_rows]

    widths = []
    for col_idx, col in enumerate(header):
        values = [col]
        values.extend(row[col_idx] if col_idx < len(row) else "" for row in display_body)
        width = min(max(len(str(v)) for v in values), max_col_width)
        widths.append(width)

    if title:
        print()
        print(title)

    print(f"[table] {csv_path}")

    def fmt(row):
        cells = []
        for idx, width in enumerate(widths):
            value = row[idx] if idx < len(row) else ""
            cells.append(_shorten(value, width).ljust(width))
        return " | ".join(cells)

    print(fmt(header))
    print("-+-".join("-" * width for width in widths))

    for row in display_body:
        print(fmt(row))

    if len(body) > max_rows:
        print(f"... ({len(body) - max_rows} more rows)")
