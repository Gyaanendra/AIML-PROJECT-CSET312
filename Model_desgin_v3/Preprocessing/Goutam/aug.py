# aug_multi_thread_concurrent.py
#
# ⚠️  FOR TRUE CONCURRENCY start Ollama with:
#       OLLAMA_NUM_PARALLEL=4 ollama serve
#     WORKERS_PER_FILE=2 × 2 files = 4 total in-flight requests.

import os
import sys
import re
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import ollama
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

# ────────────────────────────────────────────────
#              CONFIGURATION
# ────────────────────────────────────────────────

FOLDER      = r"./"
REAL_INPUT  = os.path.join(FOLDER, "real_part_2_rows_30001_to_60000.csv")
FAKE_INPUT  = os.path.join(FOLDER, "fake_part_2_rows_30001_to_60000.csv")
REAL_OUTPUT = os.path.join(FOLDER, "real_120k_lfm_part_2.csv")
FAKE_OUTPUT = os.path.join(FOLDER, "fake_120k_lfm_part_2.csv")

TEXT_COLUMN  = "text"
LABEL_COLUMN = "label"
MODEL        = "tomng/lfm2.5-instruct:latest"

WORKERS_PER_FILE = 2      # × 2 files = 4 total → must equal OLLAMA_NUM_PARALLEL
CHECKPOINT_EVERY = 2
MIN_WORDS        = 25    # reject output below this
MAX_WORDS        = 350   # truncate output above this
# ↑ Rows that can be lost on crash = up to CHECKPOINT_EVERY - 1
# Resume always restarts from the last flushed checkpoint.
# Lower = safer but more disk I/O. Set to 1 for zero loss (slowest).
REFRESH_RATE     = 4      # dashboard redraws per second

# ────────────────────────────────────────────────

TONES = [
    "positive", "negative", "neutral", "sarcastic", "angry",
    "excited", "calm", "optimistic", "pessimistic", "humorous",
    "formal", "informal", "ironic", "critical",
]

console = Console()


# ── Shared state ─────────────────────────────────────────────────────────────

class FileState:
    """All counters are thread-safe. Updated live by worker threads."""

    def __init__(self, label: str, total: int, already_done: int):
        self.label        = label
        self.total        = total          # rows in source file
        self.already_done = already_done   # rows skipped on resume

        # Counters updated every completed row
        self.processed = 0   # rows finished by workers this run (in memory)
        self.buffered  = 0   # rows sitting in buffer (not yet flushed)
        self.on_disk   = already_done  # rows confirmed written to disk

        self.failed    = 0
        self.done      = False
        self._lock     = threading.Lock()
        self.start_time = time.time()
        self.last_tones: list[str] = []

    # ── Mutators (all thread-safe) ────────────────────────────────────

    def row_completed(self, tone: str):
        """Call once per finished row — before buffering."""
        with self._lock:
            self.processed += 1
            self.buffered  += 1
            self.last_tones = (self.last_tones + [tone])[-5:]

    def rows_flushed(self, n: int):
        """Call after each CSV flush — moves n rows from buffered → on_disk."""
        with self._lock:
            self.buffered = max(0, self.buffered - n)
            self.on_disk += n

    def row_failed(self):
        with self._lock:
            self.failed += 1

    # ── Derived properties ────────────────────────────────────────────

    @property
    def total_processed(self) -> int:
        return self.already_done + self.processed

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.total_processed)

    @property
    def pct_processed(self) -> float:
        return self.total_processed / self.total * 100 if self.total else 100.0

    @property
    def pct_saved(self) -> float:
        return self.on_disk / self.total * 100 if self.total else 100.0

    @property
    def rows_per_sec(self) -> float:
        elapsed = time.time() - self.start_time
        return self.processed / elapsed if elapsed > 0 else 0.0

    @property
    def eta_str(self) -> str:
        rps = self.rows_per_sec
        if rps <= 0 or self.done:
            return "done" if self.done else "–"
        secs = self.remaining / rps
        h, r = divmod(int(secs), 3600)
        m, s = divmod(r, 60)
        if h:   return f"{h}h {m}m"
        if m:   return f"{m}m {s}s"
        return f"{s}s"


# ── Dashboard renderer ────────────────────────────────────────────────────────

def pbar(pct: float, width: int = 24, color: str = "cyan") -> str:
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] [bold]{pct:5.1f}%[/bold]"


def build_dashboard(states: list[FileState], start_time: float) -> Panel:
    elapsed = time.time() - start_time
    h, r    = divmod(int(elapsed), 3600)
    m, s    = divmod(r, 60)
    elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"

    file_panels = []
    for st in states:
        color  = "green"  if st.done else "cyan"
        status = "[green]✅ DONE[/green]" if st.done else "[cyan]⚙  running[/cyan]"
        tones  = "  ".join(f"[dim]{t}[/dim]" for t in st.last_tones) or "[dim]–[/dim]"

        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        t.add_column(style="bold", width=12)
        t.add_column(min_width=32)

        t.add_row("Status",    status)
        t.add_row("",          "")
        t.add_row("Processed", f"[bold cyan]{st.total_processed:>7,}[/bold cyan] / {st.total:,}")
        t.add_row("",          pbar(st.pct_processed, color="cyan"))
        t.add_row("",          "")
        t.add_row("Saved",     f"[bold green]{st.on_disk:>7,}[/bold green] / {st.total:,}")
        t.add_row("",          pbar(st.pct_saved, color="green"))
        t.add_row("",          "")
        t.add_row("In buffer", f"[yellow]{st.buffered:,}[/yellow] rows pending flush")
        t.add_row("Remaining", f"{st.remaining:,}")
        t.add_row("Failed",    f"[red]{st.failed}[/red]" if st.failed else "[dim]0[/dim]")
        t.add_row("Speed",     f"{st.rows_per_sec:.2f} rows/s")
        t.add_row("ETA",       st.eta_str)
        t.add_row("Tones",     tones)

        file_panels.append(
            Panel(t, title=f"[bold]{st.label.upper()}[/bold]",
                  border_style=color, expand=True)
        )

    # Overall panel
    grand_total  = sum(st.total          for st in states)
    total_proc   = sum(st.total_processed for st in states)
    total_saved  = sum(st.on_disk        for st in states)
    overall_proc = total_proc  / grand_total * 100 if grand_total else 0
    overall_save = total_saved / grand_total * 100 if grand_total else 0

    ot = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    ot.add_column(style="bold yellow", width=14)
    ot.add_column(min_width=32)
    ot.add_row("⏱  Elapsed",   elapsed_str)
    ot.add_row("", "")
    ot.add_row("📊 Processed", f"[bold cyan]{total_proc:,}[/bold cyan] / {grand_total:,}")
    ot.add_row("",             pbar(overall_proc, width=30, color="cyan"))
    ot.add_row("", "")
    ot.add_row("💾 Saved",     f"[bold green]{total_saved:,}[/bold green] / {grand_total:,}")
    ot.add_row("",             pbar(overall_save, width=30, color="green"))

    file_panels.append(
        Panel(ot, title="[bold yellow]OVERALL[/bold yellow]",
              border_style="yellow", expand=True)
    )

    return Panel(
        Columns(file_panels, equal=True, expand=True),
        title="[bold white]  Augmentation Dashboard  [/bold white]",
        border_style="white",
    )


# ── Core logic ────────────────────────────────────────────────────────────────

def create_prompt(text: str, tone: str) -> str:
    return f"""You are a neutral news rephraser. /no_think
Rewrite the following news article in a clearly {tone} tone.
Rules — you MUST follow ALL of them:
1. Keep EVERY fact, name, number, date, location, event 100% unchanged
2. Do NOT add new information
3. Do NOT remove any information
4. Do NOT change whether the story is real or fake
5. Change only wording, sentence structure and emotional framing to match the {tone} tone
6. Your response MUST be between {MIN_WORDS} and {MAX_WORDS} words — no more, no less
7. ALWAYS write complete sentences — never cut off mid-sentence
8. Do NOT think, reason, or explain — output ONLY the rewritten article text, nothing else

Article:
{text}"""


def augment_one(row: dict, label_hint: str) -> dict | None:
    orig_text = str(row.get(TEXT_COLUMN, "")).strip()
    if not orig_text:
        return None

    label = str(row.get(LABEL_COLUMN, label_hint)).strip()
    tone  = random.choice(TONES)

    try:
        resp = ollama.generate(
            model=MODEL,
            prompt=create_prompt(orig_text, tone),
            options={
                "temperature": 0.70,
                "top_p":       0.9,
                "num_predict": 1024,   # generous ceiling — 350 words * ~1.4 tok + headroom
                "num_ctx":     4096,   # large ctx so prompt + output never gets squeezed
            },
        )
        new_text = resp["response"].strip()

        # ── Aggressive cleaning ───────────────────────────────────────
        # 1. Strip <think>…</think> blocks (including unclosed ones)
        new_text = re.sub(r"<think>.*?</think>", "", new_text, flags=re.DOTALL)
        # 2. Strip orphan opening <think> tag with everything after (unclosed)
        new_text = re.sub(r"<think>.*",          "", new_text, flags=re.DOTALL)
        # 3. Strip any other XML-style tags
        new_text = re.sub(r"<[^>]+>",            "", new_text)
        # 4. Strip lines that look like meta-commentary / preamble
        lines = new_text.splitlines()
        clean_lines = [
            l for l in lines
            if not re.match(
                r"^\s*(okay|alright|sure|let me|here('s| is)|note:|rewritten|output:|article:)",
                l, re.IGNORECASE
            )
        ]
        new_text = " ".join(" ".join(clean_lines).split()).strip()
        # ─────────────────────────────────────────────────────────────

        # Enforce word count: MIN_WORDS–MAX_WORDS
        words = new_text.split()
        if len(words) < MIN_WORDS:
            new_text = "[rejected: too short]"
        elif len(words) > MAX_WORDS:
            # Truncate to MAX_WORDS, snap back to last complete sentence
            truncated = " ".join(words[:MAX_WORDS])
            last_stop = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
            new_text = truncated[:last_stop + 1] if last_stop > 0 else truncated
    except Exception as e:
        new_text = f"[generation failed: {e}]"

    return {
        "original_text":  orig_text,
        "label":          label,
        "tone":           tone,
        "augmented_text": new_text,
        "source_file":    os.path.basename(str(row.get("source_file", ""))),
        "model":          MODEL,
    }


def flush_buffer(buffer: list, output_path: str) -> int:
    """Write buffer to CSV, clear it. Returns rows written. Caller holds lock."""
    if not buffer:
        return 0
    n = len(buffer)
    write_header = not os.path.exists(output_path)
    pd.DataFrame(buffer).to_csv(
        output_path, mode="a", index=False,
        header=write_header, encoding="utf-8",
    )
    buffer.clear()
    return n


def augment_file(input_path: str, output_path: str,
                 label_hint: str, workers: int,
                 state: FileState) -> bool:

    df = pd.read_csv(input_path)
    if state.already_done:
        df = df.iloc[state.already_done:].reset_index(drop=True)

    if df.empty:
        state.done = True
        return True

    buffer   = []
    buf_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(augment_one, row.to_dict(), label_hint): idx
            for idx, row in df.iterrows()
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    # ① Update processed + buffered counter immediately
                    state.row_completed(result["tone"])

                    with buf_lock:
                        buffer.append(result)
                        # ② Flush every CHECKPOINT_EVERY rows → update on_disk
                        if len(buffer) >= CHECKPOINT_EVERY:
                            n = flush_buffer(buffer, output_path)
                            state.rows_flushed(n)
                else:
                    state.row_failed()
            except Exception:
                state.row_failed()

    # Final flush
    with buf_lock:
        n = flush_buffer(buffer, output_path)
        if n:
            state.rows_flushed(n)

    state.done = True
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def count_existing(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


if __name__ == "__main__":
    try:
        ollama.show(MODEL)
        console.print(f"[green]✓[/green] Model ready: [bold]{MODEL}[/bold]")
    except Exception:
        console.print(f"[red]Run first:[/red]  ollama pull {MODEL}")
        sys.exit(1)

    for path in (REAL_INPUT, FAKE_INPUT):
        if not os.path.exists(path):
            console.print(f"[red]❌ File not found: {path}[/red]")
            sys.exit(1)

    real_total = len(pd.read_csv(REAL_INPUT))
    fake_total = len(pd.read_csv(FAKE_INPUT))
    real_done  = count_existing(REAL_OUTPUT)
    fake_done  = count_existing(FAKE_OUTPUT)

    total_workers = WORKERS_PER_FILE * 2
    console.print(f"\n[bold]Processing real + fake CONCURRENTLY[/bold]")
    console.print(f"  {WORKERS_PER_FILE} workers × 2 files = {total_workers} total in-flight")
    console.print(f"  [yellow]⚠[/yellow]  Ollama must be running with [bold]OLLAMA_NUM_PARALLEL={total_workers}[/bold]")
    if real_done:
        console.print(f"  [yellow]↩  REAL:[/yellow] resuming from row {real_done:,} / {real_total:,}")
    if fake_done:
        console.print(f"  [yellow]↩  FAKE:[/yellow] resuming from row {fake_done:,} / {fake_total:,}")
    console.print()

    real_state = FileState("real", real_total, real_done)
    fake_state = FileState("fake", fake_total, fake_done)
    start_time = time.time()
    results: dict[str, bool] = {}

    with Live(build_dashboard([real_state, fake_state], start_time),
              refresh_per_second=REFRESH_RATE, console=console) as live:

        with ThreadPoolExecutor(max_workers=2) as file_pool:
            fut_real = file_pool.submit(
                augment_file, REAL_INPUT, REAL_OUTPUT, "real", WORKERS_PER_FILE, real_state
            )
            fut_fake = file_pool.submit(
                augment_file, FAKE_INPUT, FAKE_OUTPUT, "fake", WORKERS_PER_FILE, fake_state
            )

            while not (fut_real.done() and fut_fake.done()):
                live.update(build_dashboard([real_state, fake_state], start_time))
                time.sleep(1 / REFRESH_RATE)

            live.update(build_dashboard([real_state, fake_state], start_time))
            results["real"] = fut_real.result()
            results["fake"] = fut_fake.result()

    elapsed = time.time() - start_time
    h, r    = divmod(int(elapsed), 3600)
    m, s    = divmod(r, 60)

    console.print()
    console.rule("[bold white]FINAL SUMMARY[/bold white]")
    console.print(f"  real  → [{'green' if results['real'] else 'red'}]{'✅ OK' if results['real'] else '❌ FAILED'}[/]"
                  f"   processed: {real_state.total_processed:,}   saved: {real_state.on_disk:,} / {real_total:,}")
    console.print(f"  fake  → [{'green' if results['fake'] else 'red'}]{'✅ OK' if results['fake'] else '❌ FAILED'}[/]"
                  f"   processed: {fake_state.total_processed:,}   saved: {fake_state.on_disk:,} / {fake_total:,}")
    console.print(f"  ⏱  Total time: {h:02d}:{m:02d}:{s:02d}")
    console.rule()