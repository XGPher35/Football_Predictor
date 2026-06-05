import argparse
import re
from datetime import datetime, timezone
from typing import List, Dict
import difflib

import pandas as pd
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from nn_football.data import read_epl_csvs
from nn_football.predict import predict_match


console = Console()


def suggest_teams(teams: List[str], query: str, limit: int = 8) -> List[str]:
    q = query.strip().lower()
    if not q:
        return teams[:limit]
    starts = [t for t in teams if t.lower().startswith(q)]
    contains = [t for t in teams if q in t.lower() and t not in starts]
    return (starts + contains)[:limit]


def normalize_name(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"[\.'’]", "", n)
    n = re.sub(r"\b(fc|afc|cfc|cf)\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def build_aliases(teams: List[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for t in teams:
        base = normalize_name(t)
        aliases[base] = t
        aliases[base.replace("-", " ")] = t
        if "man united" in base or "man utd" in base:
            aliases["manchester united"] = t
            aliases["man united"] = t
            aliases["man utd"] = t
        if "man city" in base or "manchester city" in base:
            aliases["manchester city"] = t
            aliases["man city"] = t
        if "nottingham forest" in base or "nottingham" in base:
            aliases["nottingham forest"] = t
            aliases["nottm forest"] = t
            aliases["notts forest"] = t
        if "tottenham" in base:
            aliases["spurs"] = t
            aliases["tottenham"] = t
    return aliases


def fuzzy_team_match(
    teams: List[str], aliases: Dict[str, str], query: str
) -> List[str]:
    q = normalize_name(query)
    if q in aliases:
        return [aliases[q]]

    candidates = list(aliases.keys())
    close = difflib.get_close_matches(q, candidates, n=5, cutoff=0.55)
    results = []
    for c in close:
        team = aliases.get(c)
        if team and team not in results:
            results.append(team)
    if results:
        return results

    return suggest_teams(teams, query)


def prompt_team(label: str, teams: List[str], aliases: Dict[str, str]) -> str:
    while True:
        name = Prompt.ask(label).strip()
        matches = fuzzy_team_match(teams, aliases, name)
        if name in teams:
            return name
        norm = normalize_name(name)
        if norm in aliases:
            return aliases[norm]
        if matches:
            console.print("Did you mean:")
            for i, t in enumerate(matches, start=1):
                console.print(f"  {i}. {t}")
            pick = Prompt.ask("Pick a number or press Enter to retype", default="")
            if pick.isdigit():
                idx = int(pick) - 1
                if 0 <= idx < len(matches):
                    return matches[idx]
        console.print("Please enter a valid team name from the dataset.")


def prompt_date() -> pd.Timestamp:
    while True:
        date_str = Prompt.ask(
            "Match date (YYYY-MM-DD)",
            default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )
        try:
            return pd.to_datetime(date_str)
        except Exception:
            console.print("Invalid date format. Try again.")


def get_recent_form(
    df: pd.DataFrame,
    team: str,
    match_date: pd.Timestamp,
    n: int = 5,
) -> pd.DataFrame:
    df = df[df["Date"] < match_date].copy()
    is_home = df["HomeTeam"] == team
    is_away = df["AwayTeam"] == team
    team_df = df[is_home | is_away].copy()
    team_df = team_df.sort_values("Date").tail(n)

    rows = []
    for _, r in team_df.iterrows():
        home = r["HomeTeam"]
        away = r["AwayTeam"]
        gf = int(r["FTHG"]) if team == home else int(r["FTAG"])
        ga = int(r["FTAG"]) if team == home else int(r["FTHG"])
        if gf > ga:
            res = "W"
        elif gf < ga:
            res = "L"
        else:
            res = "D"
        rows.append(
            {
                "date": r["Date"].strftime("%Y-%m-%d"),
                "home": home,
                "away": away,
                "gf": gf,
                "ga": ga,
                "res": res,
            }
        )
    return pd.DataFrame(rows)


def render_form_table(team: str, form_df: pd.DataFrame) -> None:
    table = Table(title=f"Last {len(form_df)} matches for {team}")
    table.add_column("Date", justify="left")
    table.add_column("Home (G)", justify="left")
    table.add_column("Away (G)", justify="left")
    table.add_column("Res", justify="center")

    for _, r in form_df.iterrows():
        home = f"{r['home']} ({r['gf'] if r['home'] == team else r['ga']})"
        away = f"{r['away']} ({r['ga'] if r['home'] == team else r['gf']})"
        table.add_row(
            r["date"],
            home,
            away,
            r["res"],
        )
    console.print(table)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/epl")
    p.add_argument("--ckpt_path", type=str, default="artifacts/model_best.pt")
    p.add_argument("--scaler_path", type=str, default="artifacts/scalers.pkl")
    p.add_argument("--seq_len", type=int, default=5)
    args = p.parse_args()

    df = read_epl_csvs(args.data_dir)
    teams = sorted(df["HomeTeam"].unique().tolist())
    aliases = build_aliases(teams)

    console.print("Football match predictor (dataset-only)")
    home = prompt_team("Home team", teams, aliases)
    away = prompt_team("Away team", teams, aliases)
    match_date = prompt_date()

    home_form = get_recent_form(df, home, match_date, n=args.seq_len)
    away_form = get_recent_form(df, away, match_date, n=args.seq_len)
    if len(home_form) > 0:
        render_form_table(home, home_form)
    if len(away_form) > 0:
        render_form_table(away, away_form)

    pred = predict_match(
        df,
        home_team=home,
        away_team=away,
        match_date=match_date,
        ckpt_path=args.ckpt_path,
        scaler_path=args.scaler_path,
        seq_len=args.seq_len,
    )

    console.print(
        f"Outcome probs — H: {pred['probs']['H']:.3f}, D: {pred['probs']['D']:.3f}, A: {pred['probs']['A']:.3f}"
    )
    home_goals = int(round(home_goals_unrounded := pred["goals"]["home"]))
    away_goals = int(round(away_goals_unrounded := pred["goals"]["away"]))
    console.print(
        f"Goals — {home} ({home_goals} [{home_goals_unrounded:.1f}]) vs {away} ({away_goals} [{away_goals_unrounded:.1f}])"
    )
    console.print(
        f"Shots — home: {pred['stats']['home_shots']:.1f}, away: {pred['stats']['away_shots']:.1f}"
    )
    console.print(
        f"Corners — home: {pred['stats']['home_corners']:.1f}, away: {pred['stats']['away_corners']:.1f}"
    )


if __name__ == "__main__":
    main()
