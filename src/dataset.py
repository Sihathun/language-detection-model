# src/dataset.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_PROC = "data/processed"

class LangDataset(Dataset):
    def __init__(self, split="train", test_frac=0.2, val_frac=0.1, load_path=DATA_PROC):
        """
        Loads X.npy and y.npy from data/processed.
        Performs a deterministic split (train/val/test).
        """
        xp = np.load(os.path.join(load_path, "X.npy"))
        yp = np.load(os.path.join(load_path, "y.npy"))

        # deterministic shuffle
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(xp))
        xp = xp[perm]
        yp = yp[perm]

        n = len(xp)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_test - n_val

        train_X = xp[:n_train]
        train_y = yp[:n_train]
        val_X = xp[n_train:n_train+n_val]
        val_y = yp[n_train:n_train+n_val]
        test_X = xp[n_train+n_val:]
        test_y = yp[n_train+n_val:]

        if split == "train":
            self.X, self.y = train_X, train_y
        elif split == "val":
            self.X, self.y = val_X, val_y
        elif split == "test":
            self.X, self.y = test_X, test_y
        else:
            raise ValueError("split must be one of train/val/test")

        # load vocab and label_map if present
        vocab_path = os.path.join(load_path, "vocab.json")
        label_path = os.path.join(load_path, "label_map.json")
        self.idx2char = []
        self.label2id = {}
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.idx2char = json.load(f)
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                self.label2id = json.load(f)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.long)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y
