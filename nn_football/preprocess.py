import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler


def compute_team_elo(df: pd.DataFrame, k: float = 20.0) -> pd.DataFrame:
    """Compute pre-match Elo ratings for home and away teams.

    Returns a DataFrame with added columns `EloHome_pre` and `EloAway_pre`.
    """
    df = df.sort_values("Date").copy()
    ratings: Dict[str, float] = {}
    elo_home = []
    elo_away = []
    for _, r in df.iterrows():
        home = r["HomeTeam"]
        away = r["AwayTeam"]
        Rh = ratings.get(home, 1500.0)
        Ra = ratings.get(away, 1500.0)
        elo_home.append(Rh)
        elo_away.append(Ra)

        # update after match
        gh = int(r["FTHG"]) if pd.notna(r["FTHG"]) else 0
        ga = int(r["FTAG"]) if pd.notna(r["FTAG"]) else 0
        Eh = 1.0 / (1.0 + 10 ** ((Ra - Rh) / 400.0))
        if gh > ga:
            Sh = 1.0
            Sa = 0.0
        elif gh < ga:
            Sh = 0.0
            Sa = 1.0
        else:
            Sh = 0.5
            Sa = 0.5
        ratings[home] = Rh + k * (Sh - Eh)
        ratings[away] = Ra + k * (Sa - (1 - Eh))

    df["EloHome_pre"] = elo_home
    df["EloAway_pre"] = elo_away
    return df


def build_dataset(
    df: pd.DataFrame, seq_len: int = 5
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, object],
    np.ndarray,
]:
    """Build numpy arrays for model training.

    Returns: homes, aways, stat_feats, outcome_labels, goals_labels, stats_labels
    """
    df = compute_team_elo(df)
    df = df.sort_values("Date").reset_index(drop=True)

    # history per team
    history: Dict[str, List[Dict]] = {}
    ref_hist: Dict[str, Dict[str, float]] = {}
    ref_global = {"count": 0.0, "home_win": 0.0, "cards": 0.0}
    homes = []
    aways = []
    stat_feats = []
    outcome = []
    goals = []
    stats = []
    sample_dates = []

    for _, row in df.iterrows():
        h = row["HomeTeam"]
        a = row["AwayTeam"]
        history.setdefault(h, [])
        history.setdefault(a, [])

        # only include matches where both teams have at least seq_len previous matches
        if len(history[h]) < seq_len or len(history[a]) < seq_len:
            # append current match to histories and continue
            history[h].append(
                {
                    "GF": int(row["FTHG"]),
                    "GA": int(row["FTAG"]),
                    "SF": int(row.get("HS", 0)),
                    "SA": int(row.get("AS", 0)),
                    "SOTF": int(row.get("HST", 0)),
                    "SOTA": int(row.get("AST", 0)),
                    "CF": int(row.get("HC", 0)),
                    "CA": int(row.get("AC", 0)),
                    "FF": int(row.get("HF", 0)),
                    "FA": int(row.get("AF", 0)),
                    "HTGF": int(row.get("HTHG", 0)),
                    "HTGA": int(row.get("HTAG", 0)),
                    "HTR": _ht_result(row.get("HTR"), is_home=True),
                    "YF": int(row.get("HY", 0)),
                    "YA": int(row.get("AY", 0)),
                    "RF": int(row.get("HR", 0)),
                    "RA": int(row.get("AR", 0)),
                    "date": row["Date"],
                }
            )
            history[a].append(
                {
                    "GF": int(row["FTAG"]),
                    "GA": int(row["FTHG"]),
                    "SF": int(row.get("AS", 0)),
                    "SA": int(row.get("HS", 0)),
                    "SOTF": int(row.get("AST", 0)),
                    "SOTA": int(row.get("HST", 0)),
                    "CF": int(row.get("AC", 0)),
                    "CA": int(row.get("HC", 0)),
                    "FF": int(row.get("AF", 0)),
                    "FA": int(row.get("HF", 0)),
                    "HTGF": int(row.get("HTAG", 0)),
                    "HTGA": int(row.get("HTHG", 0)),
                    "HTR": _ht_result(row.get("HTR"), is_home=False),
                    "YF": int(row.get("AY", 0)),
                    "YA": int(row.get("HY", 0)),
                    "RF": int(row.get("AR", 0)),
                    "RA": int(row.get("HR", 0)),
                    "date": row["Date"],
                }
            )
            _update_ref_stats(ref_hist, ref_global, row)
            continue

        def seq_array(team_hist):
            seq = team_hist[-seq_len:]
            arr = np.zeros((seq_len, 17), dtype=np.float32)
            for i, m in enumerate(reversed(seq)):
                arr[i, 0] = m.get("GF", 0)
                arr[i, 1] = m.get("GA", 0)
                arr[i, 2] = m.get("SF", 0)
                arr[i, 3] = m.get("SA", 0)
                arr[i, 4] = m.get("SOTF", 0)
                arr[i, 5] = m.get("SOTA", 0)
                arr[i, 6] = m.get("CF", 0)
                arr[i, 7] = m.get("CA", 0)
                arr[i, 8] = m.get("FF", 0)
                arr[i, 9] = m.get("FA", 0)
                arr[i, 10] = m.get("HTGF", 0)
                arr[i, 11] = m.get("HTGA", 0)
                arr[i, 12] = m.get("HTR", 0)
                arr[i, 13] = m.get("YF", 0)
                arr[i, 14] = m.get("YA", 0)
                arr[i, 15] = m.get("RF", 0)
                arr[i, 16] = m.get("RA", 0)
            return arr

        homes.append(seq_array(history[h]))
        aways.append(seq_array(history[a]))

        # static features: elo diff, rest days (last match gap), home_is_home
        h_rest = (row["Date"] - history[h][-1]["date"]).days if history[h] else 999
        a_rest = (row["Date"] - history[a][-1]["date"]).days if history[a] else 999
        elo_diff = float(
            row.get("EloHome_pre", 1500.0) - row.get("EloAway_pre", 1500.0)
        )
        ref_home_win, ref_cards = _ref_features(
            ref_hist, ref_global, row.get("Referee")
        )
        stat_feats.append(
            [
                elo_diff,
                np.log1p(h_rest),
                np.log1p(a_rest),
                1.0,
                ref_home_win,
                ref_cards,
            ]
        )

        # labels
        ftr = row["FTR"]
        outcome.append({"H": 0, "D": 1, "A": 2}.get(ftr, 1))
        goals.append([int(row["FTHG"]), int(row["FTAG"])])
        stats.append(
            [
                int(row.get("HS", 0)),
                int(row.get("AS", 0)),
                int(row.get("HST", 0)),
                int(row.get("AST", 0)),
                int(row.get("HC", 0)),
                int(row.get("AC", 0)),
                int(row.get("HF", 0)),
                int(row.get("AF", 0)),
                int(row.get("HY", 0)),
                int(row.get("AY", 0)),
                int(row.get("HR", 0)),
                int(row.get("AR", 0)),
                int(row.get("HTHG", 0)),
                int(row.get("HTAG", 0)),
            ]
        )
        sample_dates.append(row["Date"])

        # update histories
        history[h].append(
            {
                "GF": int(row["FTHG"]),
                "GA": int(row["FTAG"]),
                "SF": int(row.get("HS", 0)),
                "SA": int(row.get("AS", 0)),
                "SOTF": int(row.get("HST", 0)),
                "SOTA": int(row.get("AST", 0)),
                "CF": int(row.get("HC", 0)),
                "CA": int(row.get("AC", 0)),
                "FF": int(row.get("HF", 0)),
                "FA": int(row.get("AF", 0)),
                "HTGF": int(row.get("HTHG", 0)),
                "HTGA": int(row.get("HTAG", 0)),
                "HTR": _ht_result(row.get("HTR"), is_home=True),
                "YF": int(row.get("HY", 0)),
                "YA": int(row.get("AY", 0)),
                "RF": int(row.get("HR", 0)),
                "RA": int(row.get("AR", 0)),
                "date": row["Date"],
            }
        )
        history[a].append(
            {
                "GF": int(row["FTAG"]),
                "GA": int(row["FTHG"]),
                "SF": int(row.get("AS", 0)),
                "SA": int(row.get("HS", 0)),
                "SOTF": int(row.get("AST", 0)),
                "SOTA": int(row.get("HST", 0)),
                "CF": int(row.get("AC", 0)),
                "CA": int(row.get("HC", 0)),
                "FF": int(row.get("AF", 0)),
                "FA": int(row.get("HF", 0)),
                "HTGF": int(row.get("HTAG", 0)),
                "HTGA": int(row.get("HTHG", 0)),
                "HTR": _ht_result(row.get("HTR"), is_home=False),
                "YF": int(row.get("AY", 0)),
                "YA": int(row.get("HY", 0)),
                "RF": int(row.get("AR", 0)),
                "RA": int(row.get("HR", 0)),
                "date": row["Date"],
            }
        )
        _update_ref_stats(ref_hist, ref_global, row)

    homes = np.stack(homes) if homes else np.zeros((0, seq_len, 17), dtype=np.float32)
    aways = np.stack(aways) if aways else np.zeros((0, seq_len, 17), dtype=np.float32)
    stat_feats = np.array(stat_feats, dtype=np.float32)
    outcome = np.array(outcome, dtype=np.int64)
    goals = np.log1p(np.array(goals, dtype=np.float32))
    stats = np.log1p(np.array(stats, dtype=np.float32))
    sample_dates = np.array(sample_dates, dtype="datetime64[ns]")

    # scale numeric static features
    seq_scaler = StandardScaler()
    static_scaler = StandardScaler()
    if homes.shape[0] > 0:
        flat = np.concatenate(
            [homes.reshape(-1, homes.shape[2]), aways.reshape(-1, aways.shape[2])],
            axis=0,
        )
        seq_scaler.fit(flat)
        homes = seq_scaler.transform(homes.reshape(-1, homes.shape[2])).reshape(
            homes.shape
        )
        aways = seq_scaler.transform(aways.reshape(-1, aways.shape[2])).reshape(
            aways.shape
        )

    if len(stat_feats) > 0:
        stat_feats[:, :5] = static_scaler.fit_transform(stat_feats[:, :5])

    scalers = {
        "seq_scaler": seq_scaler,
        "static_scaler": static_scaler,
        "target_transform": "log1p",
    }

    return homes, aways, stat_feats, outcome, goals, stats, scalers, sample_dates


def _ht_result(val: Optional[str], is_home: bool) -> int:
    if val == "H":
        return 1 if is_home else -1
    if val == "A":
        return -1 if is_home else 1
    return 0


def _update_ref_stats(
    ref_hist: Dict[str, Dict[str, float]], ref_global: Dict[str, float], row
) -> None:
    ref = row.get("Referee")
    if not isinstance(ref, str) or not ref:
        return
    h_win = 1.0 if row.get("FTR") == "H" else 0.0
    cards = float(
        row.get("HY", 0) + row.get("AY", 0) + row.get("HR", 0) + row.get("AR", 0)
    )
    stats = ref_hist.setdefault(ref, {"count": 0.0, "home_win": 0.0, "cards": 0.0})
    stats["count"] += 1.0
    stats["home_win"] += h_win
    stats["cards"] += cards
    ref_global["count"] += 1.0
    ref_global["home_win"] += h_win
    ref_global["cards"] += cards


def _ref_features(
    ref_hist: Dict[str, Dict[str, float]],
    ref_global: Dict[str, float],
    referee: Optional[str],
) -> Tuple[float, float]:
    if not isinstance(referee, str) or not referee:
        return _ref_global_features(ref_global)
    stats = ref_hist.get(referee)
    if not stats or stats["count"] == 0.0:
        return _ref_global_features(ref_global)
    return stats["home_win"] / stats["count"], stats["cards"] / stats["count"]


def _ref_global_features(ref_global: Dict[str, float]) -> Tuple[float, float]:
    if ref_global["count"] == 0.0:
        return 0.0, 0.0
    return ref_global["home_win"] / ref_global["count"], ref_global[
        "cards"
    ] / ref_global["count"]


def build_single_match_features(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    seq_len: int,
    scalers: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create model features for a single match using only data before match_date."""
    df = df.sort_values("Date").reset_index(drop=True)

    history = {}
    ratings: Dict[str, float] = {}
    ref_hist: Dict[str, Dict[str, float]] = {}
    ref_global = {"count": 0.0, "home_win": 0.0, "cards": 0.0}

    for _, row in df.iterrows():
        if row["Date"] >= match_date:
            break

        h = row["HomeTeam"]
        a = row["AwayTeam"]
        history.setdefault(h, [])
        history.setdefault(a, [])

        Rh = ratings.get(h, 1500.0)
        Ra = ratings.get(a, 1500.0)

        gh = int(row["FTHG"]) if pd.notna(row["FTHG"]) else 0
        ga = int(row["FTAG"]) if pd.notna(row["FTAG"]) else 0
        Eh = 1.0 / (1.0 + 10 ** ((Ra - Rh) / 400.0))
        if gh > ga:
            Sh = 1.0
            Sa = 0.0
        elif gh < ga:
            Sh = 0.0
            Sa = 1.0
        else:
            Sh = 0.5
            Sa = 0.5
        ratings[h] = Rh + 20.0 * (Sh - Eh)
        ratings[a] = Ra + 20.0 * (Sa - (1 - Eh))

        history[h].append(
            {
                "GF": int(row["FTHG"]),
                "GA": int(row["FTAG"]),
                "SF": int(row.get("HS", 0)),
                "SA": int(row.get("AS", 0)),
                "SOTF": int(row.get("HST", 0)),
                "SOTA": int(row.get("AST", 0)),
                "CF": int(row.get("HC", 0)),
                "CA": int(row.get("AC", 0)),
                "FF": int(row.get("HF", 0)),
                "FA": int(row.get("AF", 0)),
                "HTGF": int(row.get("HTHG", 0)),
                "HTGA": int(row.get("HTAG", 0)),
                "HTR": _ht_result(row.get("HTR"), is_home=True),
                "YF": int(row.get("HY", 0)),
                "YA": int(row.get("AY", 0)),
                "RF": int(row.get("HR", 0)),
                "RA": int(row.get("AR", 0)),
                "date": row["Date"],
            }
        )
        history[a].append(
            {
                "GF": int(row["FTAG"]),
                "GA": int(row["FTHG"]),
                "SF": int(row.get("AS", 0)),
                "SA": int(row.get("HS", 0)),
                "SOTF": int(row.get("AST", 0)),
                "SOTA": int(row.get("HST", 0)),
                "CF": int(row.get("AC", 0)),
                "CA": int(row.get("HC", 0)),
                "FF": int(row.get("AF", 0)),
                "FA": int(row.get("HF", 0)),
                "HTGF": int(row.get("HTAG", 0)),
                "HTGA": int(row.get("HTHG", 0)),
                "HTR": _ht_result(row.get("HTR"), is_home=False),
                "YF": int(row.get("AY", 0)),
                "YA": int(row.get("HY", 0)),
                "RF": int(row.get("AR", 0)),
                "RA": int(row.get("HR", 0)),
                "date": row["Date"],
            }
        )
        _update_ref_stats(ref_hist, ref_global, row)

    if home_team not in history or away_team not in history:
        raise ValueError("Teams not found in history before match_date")
    if len(history[home_team]) < seq_len or len(history[away_team]) < seq_len:
        raise ValueError("Insufficient match history for one or both teams")

    def seq_array(team_hist):
        seq = team_hist[-seq_len:]
        arr = np.zeros((seq_len, 17), dtype=np.float32)
        for i, m in enumerate(reversed(seq)):
            arr[i, 0] = m.get("GF", 0)
            arr[i, 1] = m.get("GA", 0)
            arr[i, 2] = m.get("SF", 0)
            arr[i, 3] = m.get("SA", 0)
            arr[i, 4] = m.get("SOTF", 0)
            arr[i, 5] = m.get("SOTA", 0)
            arr[i, 6] = m.get("CF", 0)
            arr[i, 7] = m.get("CA", 0)
            arr[i, 8] = m.get("FF", 0)
            arr[i, 9] = m.get("FA", 0)
            arr[i, 10] = m.get("HTGF", 0)
            arr[i, 11] = m.get("HTGA", 0)
            arr[i, 12] = m.get("HTR", 0)
            arr[i, 13] = m.get("YF", 0)
            arr[i, 14] = m.get("YA", 0)
            arr[i, 15] = m.get("RF", 0)
            arr[i, 16] = m.get("RA", 0)
        return arr

    home_seq = seq_array(history[home_team])
    away_seq = seq_array(history[away_team])

    h_rest = (match_date - history[home_team][-1]["date"]).days
    a_rest = (match_date - history[away_team][-1]["date"]).days
    elo_diff = float(ratings.get(home_team, 1500.0) - ratings.get(away_team, 1500.0))
    ref_home_win, ref_cards = _ref_features(ref_hist, ref_global, None)
    stat_feats = np.array(
        [[elo_diff, np.log1p(h_rest), np.log1p(a_rest), 1.0, ref_home_win, ref_cards]],
        dtype=np.float32,
    )
    seq_scaler = scalers["seq_scaler"]
    static_scaler = scalers["static_scaler"]
    home_seq = seq_scaler.transform(home_seq.reshape(-1, home_seq.shape[1])).reshape(
        home_seq.shape
    )
    away_seq = seq_scaler.transform(away_seq.reshape(-1, away_seq.shape[1])).reshape(
        away_seq.shape
    )
    stat_feats[:, :5] = static_scaler.transform(stat_feats[:, :5])

    return home_seq, away_seq, stat_feats
