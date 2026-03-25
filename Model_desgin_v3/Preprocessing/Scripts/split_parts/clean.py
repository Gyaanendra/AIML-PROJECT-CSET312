import pandas as pd
import re
import os

FILES = [
    "real_120k_lfm.csv",
    "real_120k_lfm_part_2.csv",
    "real_120k_lfm_part3.csv",
    "real_120k_lfm_part_4.csv",
    "fake_120k_lfm.csv",
    "fake_120k_lfm_part_2.csv",
    "fake_120k_lfm_part3.csv",
    "fake_120k_lfm_part_4.csv",
]

INPUT_DIR  = "raww"
OUTPUT_DIR = "partial_clean"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        return text

    # Remove LLM preamble phrases
    text = re.sub(r"Certainly!.*?:\s*", '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"Sure!.*?:\s*", '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"Here'?s.*?:\s*", '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"Below is.*?:\s*", '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"This is.*?:\s*", '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"The following.*?:\s*", '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"I('ve| have) (rewritten|reimagined|revised|created).*?:\s*", '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"(Rewritten|Reimagined|Revised) (version|article|text).*?:\s*", '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove Source / Read More patterns
    text = re.sub(r'Source\s*:\s*\S*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Read More\.?\.*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Read More\.?\.?\.?', '', text, flags=re.IGNORECASE)

    # Remove % of readers / junk phrases
    text = re.sub(r'\d+%\s*of readers.*?\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Add your two cents\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Before It\'s News\)', '', text, flags=re.IGNORECASE)

    # Remove special/junk characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[%@#^*~`|\\<>{}]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()

    return text


for file in FILES:
    print("\n" + "="*60)
    print(f"CLEANING: {file}")
    print("="*60)

    input_path  = os.path.join(INPUT_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, file)

    try:
        df = pd.read_csv(input_path, engine="python", on_bad_lines="skip")
        print(f"   [READ]  {df.shape[0]} rows from '{input_path}'")

        for col in ["original_text", "augmented_text"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)
                print(f"   [OK]    Cleaned '{col}'")

        df.to_csv(output_path, index=False)
        print(f"   [SAVED] → '{output_path}'")

    except FileNotFoundError:
        print(f"   [ERROR] File not found: {input_path}")
    except Exception as e:
        print(f"   [ERROR] {e}")

print("\n" + "="*60)
print(f"ALL FILES SAVED TO '{OUTPUT_DIR}/'")
print("="*60)