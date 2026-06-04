"""Quick smoke test: load CSVs, build simple features, run model forward pass."""

from pathlib import Path
import numpy as np

from nn_football.data import read_epl_csvs, build_team_sequences

import torch
from nn_football.model import BiLSTMAttentionMTL


def ftr_to_idx(x: str) -> int:
    return {"H": 0, "D": 1, "A": 2}.get(x, 1)


def make_tensor(seq, seq_len=5, feat_dim=3):
    # seq is list of small dicts with keys FTHG, FTAG, HS, AS
    arr = np.zeros((seq_len, feat_dim), dtype=np.float32)
    for i, m in enumerate(reversed(seq[-seq_len:])):
        # order: goals_for, goals_against, shots_for
        arr[i, 0] = m.get("FTHG", 0)
        arr[i, 1] = m.get("FTAG", 0)
        arr[i, 2] = m.get("HS", 0)
    return arr


def main():
    data_dir = Path("data/epl")
    df = read_epl_csvs(str(data_dir))
    records = build_team_sequences(df, seq_len=5)
    if not records:
        print("No records")
        return

    # take first 64 records for smoke
    sample = records[:64]
    seq_len = 5
    feat_dim = 3
    homes = []
    aways = []
    stat = []
    labels = []
    for r in sample:
        homes.append(make_tensor(r["home_seq"], seq_len=seq_len, feat_dim=feat_dim))
        aways.append(make_tensor(r["away_seq"], seq_len=seq_len, feat_dim=feat_dim))
        stat.append(
            [
                r["static"]["home_rest_days"] / 10.0,
                r["static"]["away_rest_days"] / 10.0,
                1.0,
                0.0,
            ]
        )
        labels.append(ftr_to_idx(r["label"]["FTR"]))

    homes = torch.from_numpy(np.stack(homes))
    aways = torch.from_numpy(np.stack(aways))
    stat = torch.tensor(stat, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    if torch is None:
        print("torch not available — data shapes:")
        print("homes", homes.shape)
        print("aways", aways.shape)
        print("stat", stat.shape)
        print("labels", labels.shape)
        return

    model = BiLSTMAttentionMTL(feature_dim=feat_dim, hid_dim=32)
    model.eval()
    with torch.no_grad():
        out = model(homes, aways, stat)
    print("Outcome logits shape:", out["outcome"].shape)
    print("Goals preds shape:", out["goals"].shape)
    print("Stats preds shape:", out["stats"].shape)


if __name__ == "__main__":
    main()
