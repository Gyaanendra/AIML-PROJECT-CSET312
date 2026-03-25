# check_rows_by_index.py
#
# Inspect specific rows by index from your augmented CSV files
# Usage examples:
#   python check_rows_by_index.py real 0 42 1500
#   python check_rows_by_index.py fake 30000 59999
#   python check_rows_by_index.py both 100 500 1000 --1based

import os
import sys
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# ── Configuration ────────────────────────────────────────────────────────────

REAL_CSV  = "real_120k_lfm_part_2.csv"
FAKE_CSV  = "fake_120k_lfm_part_2.csv"

FILES = {
    "real":  REAL_CSV,
    "fake":  FAKE_CSV,
    "both":  None,  # special: check both files
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_df(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        console.print(f"[red]File not found: {path}[/red]")
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8")
        df["augmented_text"] = df["augmented_text"].fillna("").astype(str).str.strip()
        return df
    except Exception as e:
        console.print(f"[red]Error reading {path}: {e}[/red]")
        return None


def show_row(df: pd.DataFrame, idx: int, file_label: str, one_based: bool = False):
    if idx < 0 or idx >= len(df):
        console.print(f"[yellow]Index {idx} out of range (0–{len(df)-1}) in {file_label}[/yellow]")
        return

    row = df.iloc[idx]

    orig   = row.get("original_text", "[missing]").strip()
    aug    = row.get("augmented_text", "[missing]").strip()
    tone   = row.get("tone", "?")
    label  = row.get("label", file_label)
    words  = len(aug.split()) if aug else 0

    status_color = "green"
    status_text  = "VALID"
    if "[rejected:" in aug:
        status_color = "yellow"
        status_text  = "REJECTED"
    elif "[generation failed:" in aug:
        status_color = "red"
        status_text  = "FAILED"
    elif not aug:
        status_color = "red"
        status_text  = "EMPTY"

    title = f"Row {idx + (1 if one_based else 0)}  |  {file_label.upper()}  |  {status_text}"
    content = Text.assemble(
        ("Tone:        ", "bold cyan"), (f"{tone}\n", ""),
        ("Label:       ", "bold cyan"), (f"{label}\n", ""),
        ("Words:       ", "bold cyan"), (f"{words}\n\n", ""),
        ("Original:\n", "bold underline"), (f"{orig}\n\n", ""),
        ("Augmented:\n", "bold underline"), (aug or "[empty]", ""),
    )

    console.print(Panel(
        content,
        title=f"[{status_color}]{title}[/{status_color}]",
        border_style=status_color,
        padding=(1, 2),
        expand=False
    ))
    console.print("─" * 100 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        console.print("[bold yellow]Usage:[/bold yellow]")
        console.print("  python check_rows_by_index.py <file> <index1> [index2 ...] [--1based]")
        console.print("  <file> = real | fake | both")
        console.print("  --1based   → treat input indices as 1-based (Excel-like)")
        console.print("\nExamples:")
        console.print("  python check_rows_by_index.py real 0 42 1500")
        console.print("  python check_rows_by_index.py fake 30000 --1based")
        console.print("  python check_rows_by_index.py both 100 500 1000")
        sys.exit(1)

    file_arg = sys.argv[1].lower()
    if file_arg not in FILES:
        console.print(f"[red]Invalid file: {file_arg} (use: real, fake, both)[/red]")
        sys.exit(1)

    one_based = "--1based" in sys.argv
    indices   = [arg for arg in sys.argv[2:] if arg != "--1based"]
    try:
        indices = [int(i) - (1 if one_based else 0) for i in indices]
    except ValueError:
        console.print("[red]All indices must be integers[/red]")
        sys.exit(1)

    files_to_check = [file_arg] if file_arg != "both" else ["real", "fake"]

    for f in files_to_check:
        path = FILES[f]
        df = load_df(path)
        if df is None:
            continue

        console.rule(f"[bold green] {f.upper()} File – {len(df):,} rows [/bold green]")

        found_any = False
        for idx in indices:
            if 0 <= idx < len(df):
                show_row(df, idx, f, one_based=one_based)
                found_any = True

        if not found_any and indices:
            console.print("[dim]None of the requested indices were in range.[/dim]")

if __name__ == "__main__":
    main()