"""
Dataset Cleaning Script for Language Detection

This script cleans malformed CSV files by:
1. Fixing multi-line entries
2. Removing invalid/empty texts
3. Validating label consistency
4. Removing duplicates
5. Generating data quality report

Usage:
    python scripts/clean_dataset.py --input data/raw/dataset.csv --output data/raw/dataset_cleaned.csv
"""

import os
import re
import argparse
import pandas as pd
import unicodedata
from collections import Counter


def normalize_text(text):
    """Normalize unicode and clean whitespace."""
    if pd.isna(text):
        return ""
    text = str(text)
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def is_valid_text(text, min_length=10, max_length=1000):
    """Check if text is valid for language detection."""
    if not text or len(text) < min_length:
        return False
    if len(text) > max_length:
        return False
    
    # Check if text has at least some alphabetic characters
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 3:
        return False
    
    # Check for excessive special characters
    special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_count / len(text) > 0.5:  # More than 50% special chars
        return False
    
    return True


def clean_dataset(input_path, output_path, min_length=10, max_length=1000):
    """
    Clean dataset by fixing malformed entries and validating data.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        min_length: Minimum text length
        max_length: Maximum text length
    
    Returns:
        Dictionary with cleaning statistics
    """
    print(f"📂 Reading dataset from: {input_path}")
    
    # Read CSV with error handling
    try:
        # First, try standard read
        df = pd.read_csv(input_path, encoding='utf-8')
    except Exception as e:
        print(f"⚠️  Standard CSV read failed: {e}")
        print("🔧 Attempting manual parsing...")
        
        # Manual parsing for malformed CSV
        rows = []
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Skip header if present
        start_idx = 1 if lines and ('text' in lines[0].lower() or 'language' in lines[0].lower()) else 0
        
        current_text = ""
        current_label = None
        
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if line contains a comma (potential text,label pair)
            if ',' in line:
                parts = line.rsplit(',', 1)  # Split from right to get last comma
                text_part = parts[0].strip()
                label_part = parts[1].strip() if len(parts) > 1 else ""
                
                # If we have accumulated text, save it
                if current_text and current_label:
                    rows.append({'text': current_text, 'label': current_label})
                
                # Start new entry
                current_text = text_part
                current_label = label_part
            else:
                # Continuation of previous text
                if current_text:
                    current_text += " " + line
                else:
                    current_text = line
        
        # Add last entry
        if current_text and current_label:
            rows.append({'text': current_text, 'label': current_label})
        
        df = pd.DataFrame(rows)
        print(f"✅ Manual parsing complete: {len(df)} entries found")
    
    initial_count = len(df)
    print(f"\n📊 Initial dataset: {initial_count} samples")
    
    # Ensure we have text and label columns
    if 'text' not in df.columns or 'label' not in df.columns:
        # Try to infer columns
        if len(df.columns) >= 2:
            df.columns = ['text', 'label'] + list(df.columns[2:])
        else:
            raise ValueError("Could not find 'text' and 'label' columns")
    
    # Step 1: Normalize text
    print("\n🔧 Step 1: Normalizing text...")
    df['text'] = df['text'].apply(normalize_text)
    df['label'] = df['label'].apply(lambda x: str(x).strip() if pd.notna(x) else "")
    
    # Step 2: Remove invalid entries
    print("🔧 Step 2: Removing invalid entries...")
    
    # Remove empty texts
    before = len(df)
    df = df[df['text'].str.len() > 0].copy()
    print(f"   - Removed {before - len(df)} empty texts")
    
    # Remove empty labels
    before = len(df)
    df = df[df['label'].str.len() > 0].copy()
    print(f"   - Removed {before - len(df)} empty labels")
    
    # Remove invalid text entries
    before = len(df)
    df['valid'] = df['text'].apply(lambda x: is_valid_text(x, min_length, max_length))
    df = df[df['valid']].drop('valid', axis=1).copy()
    print(f"   - Removed {before - len(df)} invalid texts (too short/long/malformed)")
    
    # Step 3: Remove duplicates
    print("🔧 Step 3: Removing duplicates...")
    before = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first').copy()
    print(f"   - Removed {before - len(df)} duplicate texts")
    
    # Step 4: Validate labels
    print("🔧 Step 4: Validating labels...")
    label_counts = df['label'].value_counts()
    
    # Remove labels with very few samples (likely errors)
    min_samples_per_label = 50
    valid_labels = label_counts[label_counts >= min_samples_per_label].index.tolist()
    before = len(df)
    df = df[df['label'].isin(valid_labels)].copy()
    print(f"   - Removed {before - len(df)} samples with rare labels (< {min_samples_per_label} samples)")
    
    # Step 5: Reset index
    df = df.reset_index(drop=True)
    
    # Generate statistics
    final_count = len(df)
    stats = {
        'initial_count': initial_count,
        'final_count': final_count,
        'removed_count': initial_count - final_count,
        'removal_rate': (initial_count - final_count) / initial_count * 100,
        'languages': len(df['label'].unique()),
        'label_distribution': df['label'].value_counts().to_dict(),
        'avg_text_length': df['text'].str.len().mean(),
        'min_text_length': df['text'].str.len().min(),
        'max_text_length': df['text'].str.len().max(),
    }
    
    # Save cleaned dataset
    print(f"\n💾 Saving cleaned dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Print summary
    print("\n" + "="*60)
    print("📊 CLEANING SUMMARY")
    print("="*60)
    print(f"Initial samples:     {stats['initial_count']:,}")
    print(f"Final samples:       {stats['final_count']:,}")
    print(f"Removed:             {stats['removed_count']:,} ({stats['removal_rate']:.1f}%)")
    print(f"Languages:           {stats['languages']}")
    print(f"Avg text length:     {stats['avg_text_length']:.1f} characters")
    print(f"Text length range:   {stats['min_text_length']}-{stats['max_text_length']}")
    print("\n📋 Samples per language:")
    for lang, count in sorted(stats['label_distribution'].items(), key=lambda x: -x[1])[:10]:
        print(f"   {lang:15s}: {count:,}")
    if len(stats['label_distribution']) > 10:
        print(f"   ... and {len(stats['label_distribution']) - 10} more")
    print("="*60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Clean language detection dataset')
    parser.add_argument('--input', type=str, 
                       default='data/raw/dataset.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', type=str,
                       default='data/raw/dataset_cleaned.csv',
                       help='Output CSV file path')
    parser.add_argument('--min-length', type=int, default=10,
                       help='Minimum text length')
    parser.add_argument('--max-length', type=int, default=1000,
                       help='Maximum text length')
    
    args = parser.parse_args()
    
    clean_dataset(args.input, args.output, args.min_length, args.max_length)


if __name__ == '__main__':
    main()
