# ⚽ Football Predictor UI

A web-based frontend for the **Data-Driven Football Forecasting** neural network predictor.  
It provides an interactive interface to simulate match outcomes using a trained ML model.

---

## 🚀 Setup

### 1. Install dependencies

You can use either **pip** or **uv**.

#### Using pip
```bash
pip install flask flask-cors
```

#### Using uv (recommended)
```bash
uv pip install flask flask-cors
```

Other dependencies like torch, pandas, numpy, scikit-learn, etc. should already be installed from the football-predictor backend project.

Make sure the trained model and preprocessing files are available:

```bash
../football-predictor/artifacts/model_best.pt
../football-predictor/artifacts/scalers.pkl
../football-predictor/data/epl/*.csv
```

These are required for predictions.

#### Run the server
```bash
python server.py
```

Open in browser

```bash
http://localhost:5000
```

#### Features
- Team Autocomplete — search and select teams from dataset
- Date Picker — choose match date for contextual prediction
- Outcome Probabilities — animated bars for Home / Draw / Away chances
- Predicted Scoreline — estimated goals + expected goals (xG)
- Match Statistics — shots, corners, fouls, cards, half-time goals
- Recent Form Analysis — last 5 matches before selected date

#### Architeture

```
football-ui/
├── server.py          # Flask API (wraps nn_football.predict)
├── pyproject.toml
├── README.md
└── static/
    ├── index.html     # Single-page frontend
    ├── style.css      # Dark theme UI
    └── app.js         # Client-side logic
``` 