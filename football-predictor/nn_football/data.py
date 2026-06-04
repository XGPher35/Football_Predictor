import os
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np


def read_epl_csvs(path: str) -> pd.DataFrame:
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f, parse_dates=["Date"])
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {path}")
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_ema(series: pd.Series, span: int = 5) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def initial_elo(df: pd.DataFrame, k: float = 20.0) -> Dict[str, float]:
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    return {t: 1500.0 for t in teams}


def update_elo(
    ratings: Dict[str, float],
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    k: float = 20.0,
):
    Rh = ratings.get(home, 1500.0)
    Ra = ratings.get(away, 1500.0)
    Eh = 1.0 / (1.0 + 10 ** ((Ra - Rh) / 400.0))
    Ea = 1.0 - Eh
    if home_goals > away_goals:
        Sh, Sa = 1.0, 0.0
    elif home_goals < away_goals:
        Sh, Sa = 0.0, 1.0
    else:
        Sh, Sa = 0.5, 0.5
    ratings[home] = Rh + k * (Sh - Eh)
    ratings[away] = Ra + k * (Sa - Ea)


def build_team_sequences(df: pd.DataFrame, seq_len: int = 5) -> List[Dict]:
    """Build sequences per match that contain last `seq_len` matches for home and away teams.

    Returns a list of dicts with keys: date, home_team, away_team, home_seq, away_seq, static, label
    """
    # track per-team history
    history: Dict[str, List[Dict]] = {}
    records = []
    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        for t in (home, away):
            history.setdefault(t, [])

        # construct feature vectors for previous matches
        def last_k(team: str):
            hist = history[team]
            if not hist:
                return []
            return hist[-seq_len:]

        rec = {
            "date": row["Date"],
            "home_team": home,
            "away_team": away,
            "home_seq": last_k(home),
            "away_seq": last_k(away),
            "static": {
                "home_rest_days": compute_rest_days(history.get(home, []), row["Date"]),
                "away_rest_days": compute_rest_days(history.get(away, []), row["Date"]),
                "home_is_home": 1,
            },
            "label": {
                "FTR": row["FTR"],
                "FTHG": int(row["FTHG"]),
                "FTAG": int(row["FTAG"]),
                "HS": int(row.get("HS", 0)),
                "AS": int(row.get("AS", 0)),
                "HC": int(row.get("HC", 0)),
                "AC": int(row.get("AC", 0)),
            },
        }

        # append current match to history for future matches
        history[home].append(
            {
                "date": row["Date"],
                "FTHG": int(row["FTHG"]),
                "FTAG": int(row["FTAG"]),
                "HS": int(row.get("HS", 0)),
                "AS": int(row.get("AS", 0)),
            }
        )
        history[away].append(
            {
                "date": row["Date"],
                "FTHG": int(row["FTAG"]),
                "FTAG": int(row["FTHG"]),
                "HS": int(row.get("AS", 0)),
                "AS": int(row.get("HS", 0)),
            }
        )

        records.append(rec)
    return records


def compute_rest_days(hist: List[Dict], date) -> int:
    if not hist:
        return 999
    last_date = hist[-1]["date"]
    delta = date - last_date
    return int(delta.days)
