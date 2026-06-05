"""
Lightweight Flask API that wraps the nn_football prediction backend.

Endpoints:
  GET  /api/teams              → list of available team names
  POST /api/predict            → prediction for a home vs away match
  GET  /api/team-form/<team>   → recent form for a specific team
  GET  /                       → serves the static frontend
"""

import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd

# pyrefly: ignore [missing-import]
from nn_football.data import read_epl_csvs
# pyrefly: ignore [missing-import]
from nn_football.predict import predict_match

# ── paths ──────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "football-predictor", "data", "epl")
CKPT_PATH  = os.path.join(os.path.dirname(__file__), "..", "football-predictor", "artifacts", "model_best.pt")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "football-predictor", "artifacts", "scalers.pkl")
SEQ_LEN    = 5
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ── bootstrap ──────────────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

print("Loading match data …")
DF = read_epl_csvs(DATA_DIR)
TEAMS = sorted(DF["HomeTeam"].unique().tolist())
print(f"  {len(TEAMS)} teams loaded.")


# ── static frontend ───────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


# ── API: team list ────────────────────────────────────────────────
@app.route("/api/teams", methods=["GET"])
def api_teams():
    return jsonify({"teams": TEAMS})


# ── API: recent form ──────────────────────────────────────────────
@app.route("/api/team-form/<team>", methods=["GET"])
def api_team_form(team):
    date_str = request.args.get("date")
    n = int(request.args.get("n", 5))
    match_date = pd.to_datetime(date_str) if date_str else pd.Timestamp.now()

    sub = DF[DF["Date"] < match_date].copy()
    is_home = sub["HomeTeam"] == team
    is_away = sub["AwayTeam"] == team
    team_df = sub[is_home | is_away].sort_values("Date").tail(n)

    rows = []
    for _, r in team_df.iterrows():
        home = r["HomeTeam"]
        away = r["AwayTeam"]
        gf = int(r["FTHG"]) if team == home else int(r["FTAG"])
        ga = int(r["FTAG"]) if team == home else int(r["FTHG"])
        res = "W" if gf > ga else ("L" if gf < ga else "D")
        rows.append({
            "date": r["Date"].strftime("%Y-%m-%d"),
            "home": home,
            "away": away,
            "gf": gf,
            "ga": ga,
            "result": res,
        })
    return jsonify({"team": team, "form": rows})


# ── API: predict ──────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)
    home = body.get("home_team", "")
    away = body.get("away_team", "")
    date_str = body.get("date")
    match_date = pd.to_datetime(date_str) if date_str else pd.Timestamp.now()

    if home not in TEAMS:
        return jsonify({"error": f"Unknown home team: {home}"}), 400
    if away not in TEAMS:
        return jsonify({"error": f"Unknown away team: {away}"}), 400
    if home == away:
        return jsonify({"error": "Home and away teams must be different"}), 400

    try:
        pred = predict_match(
            DF,
            home_team=home,
            away_team=away,
            match_date=match_date,
            ckpt_path=CKPT_PATH,
            scaler_path=SCALER_PATH,
            seq_len=SEQ_LEN,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "home_team": home,
        "away_team": away,
        "date": match_date.strftime("%Y-%m-%d"),
        "outcome": pred["probs"],
        "goals": pred["goals"],
        "stats": pred["stats"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
