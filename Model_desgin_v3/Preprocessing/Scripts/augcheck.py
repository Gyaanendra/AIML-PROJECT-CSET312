import re
import pandas as pd
import sys

# ── Config ────────────────────────────────────────────
FILE  = "fake_120k_lfm.csv"
ROW   = 4
# ─────────────────────────────────────────────────────

path = sys.argv[1] if len(sys.argv) > 1 else FILE
idx  = int(sys.argv[2]) if len(sys.argv) > 2 else ROW

df = pd.read_csv(path)

if idx >= len(df):
    print(f"❌ Row {idx} out of range — file has {len(df)} rows (0–{len(df)-1})")
    sys.exit(1)

row = df.iloc[idx]

def clean(text):
    if not isinstance(text, str):
        return str(text)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

orig = clean(row.get("original_text", "N/A"))
aug  = clean(row.get("augmented_text", "N/A"))

W = 60  # column width each side

print(f"\n{'═'*60}")
print(f"  FILE  : {path}")
print(f"  ROW   : {idx} / {len(df)-1}   LABEL: {row.get('label','N/A')}   TONE: {row.get('tone','N/A')}")
print(f"{'═'*60}\n")

# ── Side-by-side comparison ───────────────────────────
def wrap(text, width):
    """Hard-wrap text into lines of given width."""
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 <= width:
            line = (line + " " + word).lstrip()
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines

orig_lines = wrap(orig, W)
aug_lines  = wrap(aug,  W)

# Pad shorter side
max_len = max(len(orig_lines), len(aug_lines))
orig_lines += [""] * (max_len - len(orig_lines))
aug_lines  += [""] * (max_len - len(aug_lines))

header = f"{'ORIGINAL':<{W}}   {'AUGMENTED':<{W}}"
divider = f"{'─'*W}   {'─'*W}"

print(header)
print(divider)
for l, r in zip(orig_lines, aug_lines):
    print(f"{l:<{W}}   {r:<{W}}")

print(f"\n{'═'*60}\n")