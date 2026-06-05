import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
import torch

from nn_football.model import BiLSTMAttentionMTL
from nn_football.preprocess import build_single_match_features


def load_checkpoint(ckpt_path: str, device: str = "cpu") -> Dict:
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def load_scaler(scaler_path: str):
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def predict_match(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    ckpt_path: str,
    scaler_path: str,
    seq_len: int,
    device: str = "cpu",
) -> Dict:
    ckpt = load_checkpoint(ckpt_path, device=device)
    feature_dim = ckpt.get("feature_dim", 10)
    model = BiLSTMAttentionMTL(
        feature_dim=feature_dim,
        embed_dim=ckpt["config"]["embed_dim"],
        hid_dim=ckpt["config"]["hid_dim"],
        lstm_layers=ckpt["config"]["lstm_layers"],
        seq_dropout=ckpt["config"]["dropout"],
        static_dim=6,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scalers = load_scaler(scaler_path)
    home_seq, away_seq, stat_feats = build_single_match_features(
        df,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        seq_len=seq_len,
        scalers=scalers,
    )

    h = torch.from_numpy(home_seq).unsqueeze(0).float().to(device)
    a = torch.from_numpy(away_seq).unsqueeze(0).float().to(device)
    s = torch.from_numpy(stat_feats).float().to(device)

    with torch.no_grad():
        out = model(h, a, s)
        probs = torch.softmax(out["outcome"], dim=-1).cpu().numpy().squeeze(0)
        goals = out["goals"].cpu().numpy().squeeze(0)
        stats = out["stats"].cpu().numpy().squeeze(0)

    if scalers.get("target_transform") == "log1p":
        goals = np.expm1(goals)
        stats = np.expm1(stats)

    return {
        "probs": {"H": float(probs[0]), "D": float(probs[1]), "A": float(probs[2])},
        "goals": {"home": float(goals[0]), "away": float(goals[1])},
        "stats": {
            "home_shots": float(stats[0]),
            "away_shots": float(stats[1]),
            "home_shots_on_target": float(stats[2]),
            "away_shots_on_target": float(stats[3]),
            "home_corners": float(stats[4]),
            "away_corners": float(stats[5]),
            "home_fouls": float(stats[6]),
            "away_fouls": float(stats[7]),
            "home_yellows": float(stats[8]),
            "away_yellows": float(stats[9]),
            "home_reds": float(stats[10]),
            "away_reds": float(stats[11]),
            "home_ht_goals": float(stats[12]),
            "away_ht_goals": float(stats[13]),
        },
    }
