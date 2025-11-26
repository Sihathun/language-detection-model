# src/train.py
import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LangDataset
from model import CharCNN
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    gold = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(p.tolist())
            gold.extend(y.numpy().tolist())
    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, average="macro")
    return acc, f1, gold, preds

def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # datasets and loaders
    train_ds = LangDataset(split="train")
    val_ds = LangDataset(split="val")
    test_ds = LangDataset(split="test")
    # load vocab & label_map sizes
    vocab_path = "data/processed/vocab.json"
    label_path = "data/processed/label_map.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        idx2char = json.load(f)
    with open(label_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    vocab_size = len(idx2char)
    num_classes = len(label_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CharCNN(vocab_size=vocab_size, embed_dim=args.embed_dim, num_classes=num_classes,
                    num_filters=args.num_filters, kernel_sizes=tuple(map(int,args.kernels.split(","))), dropout=args.dropout)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x,y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n+1))
        avg_loss = total_loss / len(train_loader)
        val_acc, val_f1, _, _ = evaluate_model(model, val_loader, device)
        print(f"[E{epoch}] train_loss={avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        # save best
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "model_state": model.state_dict(),
                "vocab": idx2char,
                "label_map": label_map,
                "args": vars(args)
            }
            torch.save(ckpt, os.path.join(MODEL_DIR, "best_model.pt"))
            print(f"[INFO] Saved best_model.pt with val_acc={val_acc:.4f}")

    # final eval on test
    print("[INFO] Evaluating on test set with best model...")
    ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_acc, test_f1, gold, preds = evaluate_model(model, test_loader, device)
    print(f"[RESULT] Test acc: {test_acc:.4f} f1: {test_f1:.4f}")

    # confusion matrix
    label_idx2name = {int(v):k for k,v in ckpt["label_map"].items()}
    cm = confusion_matrix(gold, preds)
    plt.figure(figsize=(8,6))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[label_idx2name[i] for i in range(len(label_idx2name))],
                yticklabels=[label_idx2name[i] for i in range(len(label_idx2name))])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    print(f"[INFO] Confusion matrix saved to {MODEL_DIR}/confusion_matrix.png")

    # save history
    import json
    with open(os.path.join(MODEL_DIR, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("[INFO] Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--kernels", type=str, default="3,5,7", help="comma-separated kernel sizes")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
