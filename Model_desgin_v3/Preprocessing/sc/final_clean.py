import csv
import re
import sys
import unicodedata
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — change these if needed
# ─────────────────────────────────────────────
INPUT_FILE  = "dataset_final.csv"
OUTPUT_FILE = "dataset_cleaned.csv"
TEXT_COL    = "text"
# ─────────────────────────────────────────────


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t = text

    # ══════════════════════════════════════════
    # SECTION 1 — ENCODING & UNICODE FIXES
    # (word-safe: fixes invisible/malformed chars)
    # ══════════════════════════════════════════

    # 1a. Strip BOM and zero-width characters
    #     \ufeff = BOM, \u200b = zero-width space,
    #     \u200c = zero-width non-joiner, \u200d = zero-width joiner
    #     \u00ad = soft hyphen  — all invisible, cause silent bugs
    t = re.sub(r'[\ufeff\u200b\u200c\u200d\u00ad]', '', t)

    # 1b. Unicode NFC normalisation
    #     Makes accented chars consistent: e + combining accent → é (one char)
    t = unicodedata.normalize('NFC', t)

    # 1c. Decode HTML entities
    #     &amp; → &   &lt; → <   &gt; → >   &nbsp; → space   &quot; → "   &#39; → '
    html_entities = {
        '&amp;':  '&',
        '&lt;':   '<',
        '&gt;':   '>',
        '&nbsp;': ' ',
        '&quot;': '"',
        '&#39;':  "'",
        '&apos;': "'",
        '&mdash;': '-',
        '&ndash;': '-',
        '&hellip;': '...',
        '&ldquo;': '"',
        '&rdquo;': '"',
        '&lsquo;': "'",
        '&rsquo;': "'",
    }
    for entity, replacement in html_entities.items():
        t = t.replace(entity, replacement)
    # Catch any remaining numeric HTML entities e.g. &#160;
    t = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), t)
    t = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), t)

    # 1d. Smart quotes → straight quotes
    #     " " → "   ' ' → '
    t = t.replace('\u201c', '"').replace('\u201d', '"')   # " "
    t = t.replace('\u2018', "'").replace('\u2019', "'")   # ' '
    t = t.replace('\u00ab', '"').replace('\u00bb', '"')   # « »
    t = t.replace('\u2032', "'").replace('\u2033', '"')   # ′ ″

    # 1e. Normalise dashes
    #     em dash — and en dash – → plain hyphen -
    t = t.replace('\u2014', '-').replace('\u2013', '-')

    # ══════════════════════════════════════════
    # SECTION 2 — WHITESPACE & LINE ENDINGS
    # (word-safe: structural spacing only)
    # ══════════════════════════════════════════

    # 2a. Normalise line endings → \n only
    #     \r\n (Windows) and \r (old Mac) → \n
    t = t.replace('\r\n', '\n').replace('\r', '\n')

    # 2b. Tab → single space
    t = t.replace('\t', ' ')

    # 2c. Double (or more) space after sentence-ending punctuation → single space
    #     e.g. "Hello.  World" → "Hello. World"
    t = re.sub(r'([.!?])\s{2,}', r'\1 ', t)

    # ══════════════════════════════════════════
    # SECTION 3 — STRUCTURAL NOISE REMOVAL
    # (removes tags/markers/urls, not words)
    # ══════════════════════════════════════════

    # 3a. Remove [TAG] style markers
    t = re.sub(r'\[[^\]]{0,40}\]', '', t)

    # 3b. Remove (TAG) paren markers & empty parens
    t = re.sub(r'\([^)]{0,40}\)', '', t)

    # 3c. Remove bare URLs
    t = re.sub(r'https?://\S+', '', t)
    t = re.sub(r'www\.\S+', '', t)

    # 3d. Remove markdown links [text](url) → keep text only
    t = re.sub(r'\[([^\]]+)\]\(https?://\S+\)', r'\1', t)

    # 3e. Remove HTML tags
    t = re.sub(r'<[^>]+>', '', t)

    # 3f. Strip outer wrapping quotes (CSV artefact)
    #     e.g. the entire row text is wrapped in "..." or '...'
    t = t.strip()
    if (t.startswith('"') and t.endswith('"')) or \
       (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()

    # ══════════════════════════════════════════
    # SECTION 4 — SPECIAL CHARACTER REMOVAL
    # (removes non-alphanumeric noise)
    # ══════════════════════════════════════════

    # 4a. Remove all special/non-alphanumeric characters
    #     Keep: letters, digits, spaces, . , ! ? ' " - :
    t = re.sub(r"[^a-zA-Z0-9\s.,!?'\"'\-:]", ' ', t)

    # ══════════════════════════════════════════
    # SECTION 5 — PUNCTUATION NORMALISATION
    # (repeated punctuation → single character)
    # ══════════════════════════════════════════

    # 5a. Same char repeated with optional spaces between → single char
    t = re.sub(r'\.(\s*\.)+', '.', t)     # . . . → .
    t = re.sub(r',(\s*,)+',   ',', t)     # , , , → ,
    t = re.sub(r'-(\s*-)+',   '-', t)     # - - - → -
    t = re.sub(r':(\s*:)+',   ':', t)     # : : : → :
    t = re.sub(r'!(\s*!)+',   '!', t)     # ! ! ! → !
    t = re.sub(r'\?(\s*\?)+', '?', t)     # ? ? ? → ?

    # 5b. Mixed punctuation clusters → single period
    t = re.sub(r'[.,\-]{2,}', '.', t)

    # 5c. Final sweep: same char with spaces between
    t = re.sub(r'([.,\-:!?])(\s+\1)+', r'\1', t)

    # ══════════════════════════════════════════
    # SECTION 6 — CONTENT DEDUPLICATION
    # ══════════════════════════════════════════

    # 6a. Remove isolated single letters (except a, I)
    t = re.sub(r'(?<!\w)[b-hj-z](?!\w)', ' ', t, flags=re.IGNORECASE)

    # 6b. Remove lone 1-2 digit numbers
    t = re.sub(r'(?<!\w)\d{1,2}(?!\w)', ' ', t)

    # 6c. De-duplicate repeated headline at start of article
    half        = len(t) // 2
    first_half  = t[:half].strip()
    second_half = t[half:].strip()
    if first_half and len(first_half) > 40 and second_half.startswith(first_half[:60]):
        t = second_half

    # ══════════════════════════════════════════
    # SECTION 7 — BOILERPLATE REMOVAL
    # ══════════════════════════════════════════

    boilerplate = [
        r'\d*\s*of readers think this story is (Fact|Fiction)\.?',
        r'readers think this story is (Fact|Fiction)\.?',
        r'Filed under\s*:[^\n\.]*[\n\.]?',
        r'Updated\s+\d{1,2}:\d{2}\s*[ap]m[^\n]*',
        r'This article presents a remark\w*',
        r'Page \d+ of \d+',
        r'Click here to \w+[^\n\.]*',
        r'Subscribe to \w+[^\n\.]*',
        r'Read more\s*:?[^\n\.]*',
        r'Share this\s*:?[^\n\.]*',
    ]
    for pattern in boilerplate:
        t = re.sub(pattern, '', t, flags=re.IGNORECASE)

    # ══════════════════════════════════════════
    # SECTION 8 — FINAL WHITESPACE CLEANUP
    # ══════════════════════════════════════════

    t = re.sub(r'[ \t]{2,}', ' ', t)     # multiple spaces → one
    t = re.sub(r'\n{3,}', '\n\n', t)     # 3+ newlines → 2
    t = t.strip()

    return t


def word_count(text: str) -> int:
    return len(text.split())


def process(input_path: str, output_path: str):
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    total      = 0
    cleaned    = 0
    dropped    = 0
    duplicates = 0
    seen_texts = set()

    with open(input_path,  'r', encoding='utf-8', errors='replace') as fin, \
         open(output_path, 'w', encoding='utf-8', newline='')        as fout:

        reader = csv.DictReader(fin)

        if TEXT_COL not in reader.fieldnames:
            print(f"ERROR: Column '{TEXT_COL}' not found.")
            print(f"       Found columns: {reader.fieldnames}")
            sys.exit(1)

        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            original      = row[TEXT_COL]
            row[TEXT_COL] = clean_text(original)

            # Deduplicate exact rows
            if row[TEXT_COL] in seen_texts:
                duplicates += 1
                continue
            seen_texts.add(row[TEXT_COL])

            if row[TEXT_COL] != original:
                cleaned += 1

            writer.writerow(row)

            if total % 10_000 == 0:
                print(f"  Processed {total:,} rows...")

    kept = total - duplicates - dropped
    print(f"\n✅ Done!")
    print(f"   Total rows        : {total:,}")
    print(f"   Rows cleaned      : {cleaned:,}")
    print(f"   Duplicates removed: {duplicates:,}")
    print(f"   Rows kept         : {kept:,}")
    print(f"   Output saved      : {output_path}")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE
    print(f"Cleaning: {inp}  ->  {out}\n")
    process(inp, out)