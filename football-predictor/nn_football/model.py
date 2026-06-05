import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, input_dim)
        self.score = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        # x: (batch, seq, dim)
        h = torch.tanh(self.proj(x))
        scores = self.score(h).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        return pooled


class SeqEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attn = AttentionPool(hidden_dim * 2)

    def forward(self, x):
        z = self.embed(x)
        h, _ = self.lstm(z)
        pooled = self.attn(h)
        return pooled


class BiLSTMAttentionMTL(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 32,
        hid_dim: int = 64,
        lstm_layers: int = 2,
        seq_dropout: float = 0.2,
        static_dim: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hid_dim = hid_dim
        self.encoder = SeqEncoder(
            input_dim=feature_dim,
            embed_dim=embed_dim,
            hidden_dim=hid_dim,
            layers=lstm_layers,
            dropout=seq_dropout,
        )

        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
        )

        shared_dim = hid_dim * 5
        self.shared = nn.Sequential(
            nn.Linear(shared_dim, hid_dim * 2),
            nn.ReLU(),
            nn.Dropout(seq_dropout),
        )

        self.outcome_head = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Dropout(seq_dropout),
            nn.Linear(hid_dim, 3),
        )

        self.goals_head = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 2),
        )

        self.stats_head = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 14),
        )

    def forward(self, home_seq, away_seq, static_feats):
        h = self.encoder(home_seq)
        a = self.encoder(away_seq)
        s = self.static_mlp(static_feats)
        cat = torch.cat([h, a, s], dim=-1)
        shared = self.shared(cat)

        outcome_logits = self.outcome_head(shared)
        goals = self.goals_head(shared)
        stats = self.stats_head(shared)

        return {"outcome": outcome_logits, "goals": goals, "stats": stats}
