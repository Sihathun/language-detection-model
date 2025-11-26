# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=128, kernel_sizes=(3,5,7), dropout=0.3):
        """
        vocab_size: size of embedding (including PAD/UNK)
        embed_dim: embedding dimension
        num_classes: number of languages
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len)
        emb = self.embedding(x)           # (batch, seq_len, embed_dim)
        emb = emb.transpose(1,2)          # (batch, embed_dim, seq_len)
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb))        # (batch, num_filters, L_out)
            c = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outs.append(c)
        cat = torch.cat(conv_outs, dim=1)  # (batch, num_filters * len)
        cat = self.dropout(cat)
        logits = self.fc(cat)
        return logits
