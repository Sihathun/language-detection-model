# src/evaluate.py
import os
import argparse
import json
import numpy as np
import torch
from model import CharCNN

MODEL_DIR = "models"

def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt

def build_vocab_from_idx2char(idx2char):
    class V:
        def __init__(self, idx2char):
            self.idx2char = idx2char
            self.char2idx = {c:i for i,c in enumerate(idx2char)}
        def encode(self, s, max_len):
            import unicodedata
            s = unicodedata.normalize("NFKC", str(s)).lower()
            ids = [self.char2idx.get(ch,1) for ch in s[:max_len]]
            if len(ids) < max_len:
                ids = ids + [0]*(max_len-len(ids))
            return ids
    return V(idx2char)

def predict_text(text, ckpt_path=os.path.join(MODEL_DIR,"best_model.pt"), max_len=128):
    ckpt = load_ckpt(ckpt_path)
    idx2char = ckpt["vocab"]
    label_map = ckpt["label_map"]
    inv_label_map = {int(v):k for k,v in label_map.items()}
    vocab = build_vocab_from_idx2char(idx2char)
    model_state = ckpt["model_state"]
    args = ckpt.get("args", {})
    vocab_size = len(idx2char)
    num_classes = len(label_map)
    model = CharCNN(vocab_size=vocab_size, embed_dim=args.get("embed_dim",64),
                    num_classes=num_classes, num_filters=args.get("num_filters",128),
                    kernel_sizes=tuple(map(int,args.get("kernels","3,5,7").split(","))),
                    dropout=args.get("dropout",0.3))
    model.load_state_dict(model_state)
    model.eval()
    import torch
    ids = torch.tensor([vocab.encode(text, max_len)], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        idx = int(probs.argmax())
        return inv_label_map[idx], float(probs[idx]), probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Text to predict")
    parser.add_argument("--ckpt", type=str, default=os.path.join(MODEL_DIR,"best_model.pt"))
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}. Train a model first.")
    if args.text:
        lang, p, probs = predict_text(args.text, ckpt_path=args.ckpt)
        print(f"Predicted: {lang} ({p:.3f})")
    else:
        print("Provide --text 'some sample' to get prediction.")

if __name__ == "__main__":
    main()
