# src/preprocess.py
"""
Preprocess pipeline:
- Looks for CSV files in data/raw/ with columns (text,label) OR .txt files (one sample per line) named <lang>.txt
- Builds char-level vocabulary
- Encodes text -> fixed-length integer sequences
- Saves processed outputs to data/processed/
Outputs:
- X.npy (int32) shape (N, MAX_LEN)
- y.npy (int32) shape (N,)
- label_map.json (label -> id)
- vocab.json (idx -> char list)
"""

import os
import json
import argparse
from glob import glob
import numpy as np
import pandas as pd
import unicodedata

DATA_RAW = "data/raw"
DATA_PROC = "data/processed"

def ensure_dirs():
    os.makedirs(DATA_RAW, exist_ok=True)
    os.makedirs(DATA_PROC, exist_ok=True)
    os.makedirs("models", exist_ok=True)

def unicode_normalize(s):
    return unicodedata.normalize("NFKC", str(s))

def load_raw_data():
    """
    Returns DataFrame with columns ['text','label'].
    Acceptable raw formats:
      - CSV files under data/raw/ with columns text,label
      - TXT files named lang.txt containing samples line-by-line
    """
    rows = []
    # CSVs
    for csv_path in glob(os.path.join(DATA_RAW, "*.csv")):
        try:
            df = pd.read_csv(csv_path, usecols=["text", "label"])
            rows.append(df)
        except Exception:
            # try fallback: assume two columns without headers
            df = pd.read_csv(csv_path, header=None, names=["text", "label"])
            rows.append(df)
    # TXT files
    for txt_path in glob(os.path.join(DATA_RAW, "*.txt")):
        name = os.path.splitext(os.path.basename(txt_path))[0]
        # if name indicates it's a language file (e.g., en.txt), treat label as filename
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            df = pd.DataFrame({"text": lines, "label": [name]*len(lines)})
            rows.append(df)
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df['text'] = df['text'].astype(str).map(unicode_normalize)
    df['label'] = df['label'].astype(str)
    return df

def generate_tiny_sample(out_path=os.path.join(DATA_RAW,"tiny_sample.csv")):
    sample = [
        ("hello world", "en"),
        ("this is a test", "en"),
        ("bonjour le monde", "fr"),
        ("je suis étudiant", "fr"),
        ("hola mundo", "es"),
        ("buenos días", "es"),
        ("សួស្តី​ពិភពលោក", "km"),
        ("ជំរាបសួរ", "km"),
        ("こんにちは 世界", "jp"),
        ("おはよう", "jp"),
    ]
    os.makedirs(DATA_RAW, exist_ok=True)
    df = pd.DataFrame(sample, columns=["text","label"])
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[INFO] Tiny sample written to {out_path}")

def build_char_vocab(texts, min_freq=1, max_vocab=None):
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(list(t))
    items = [c for c,f in cnt.most_common() if f>=min_freq]
    if max_vocab:
        items = items[:max_vocab]
    # reserve 0 for PAD, 1 for UNK
    idx2char = ["<pad>","<unk>"] + items
    char2idx = {c:i for i,c in enumerate(idx2char)}
    return idx2char, char2idx

def encode_text(s, char2idx, max_len):
    s = s[:max_len]
    ids = [char2idx.get(ch, 1) for ch in s]  # unk->1
    if len(ids) < max_len:
        ids = ids + [0]*(max_len - len(ids))
    return ids

def main(args):
    ensure_dirs()
    df = load_raw_data()
    if df is None or df.empty:
        print("[WARN] No raw data found. Generating tiny sample for testing.")
        generate_tiny_sample()
        df = load_raw_data()
    print(f"[INFO] Loaded {len(df)} samples from raw data.")
    # Basic cleaning: drop empty
    df = df[df['text'].str.strip().astype(bool)].reset_index(drop=True)

    # Build label map
    labels = sorted(df['label'].unique().tolist())
    label2id = {l:i for i,l in enumerate(labels)}
    y = df['label'].map(label2id).astype(np.int32).values

    texts = df['text'].astype(str).tolist()
    idx2char, char2idx = build_char_vocab(texts, min_freq=args.min_freq, max_vocab=args.max_vocab)

    # encode
    X = np.array([encode_text(t, char2idx, args.max_len) for t in texts], dtype=np.int32)

    # save
    np.save(os.path.join(DATA_PROC, "X.npy"), X)
    np.save(os.path.join(DATA_PROC, "y.npy"), y)
    with open(os.path.join(DATA_PROC, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_PROC, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(idx2char, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved processed data to {DATA_PROC}: X.npy ({X.shape}), y.npy ({y.shape})")
    print(f"[INFO] Num labels: {len(labels)}. Vocab size: {len(idx2char)} (including PAD/UNK)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--min-freq", type=int, default=1, help="Min char freq")
    parser.add_argument("--max-vocab", type=int, default=None, help="Limit vocabulary size (not counting PAD/UNK)")
    args = parser.parse_args()
    main(args)
