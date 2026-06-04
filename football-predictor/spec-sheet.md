# System Specification: Multi-Task Deep Learning Football Predictor

## 1. Executive Summary
This document outlines the architecture for an unbiased, deep neural network designed to predict football match outcomes (Home Win, Draw, Away Win) alongside continuous post-match statistics (Goals, Shots, Corners, Fouls). The system utilizes a Multi-Task Learning (MTL) framework powered by Bidirectional LSTMs and Temporal Attention mechanisms, leveraging Transfer Learning across major European leagues to establish foundational football dynamics.

Data is available at ./data/epl/

## 2. Core Objectives
1.  **Outcome Classification:** Predict the Full-Time Result (FTR: H, D, A) with calibrated probabilities.
2.  **Statistical Regression:** Accurately forecast exact match statistics (FTHG, FTAG, HS, AS, HC, AC, etc.).
3.  **Zero Data Leakage:** Architect a strict temporal data pipeline ensuring predictions rely exclusively on $t-1$ (pre-match) knowledge.
4.  **Zero Bias:** No hardcoded modifiers for specific teams. All inferences are strictly mathematically derived from historical sequences.

---

## 3. Data Architecture & Feature Engineering Pipeline

### 3.1 Data Ingestion
*   **Sources:** Historical CSV datasets (EPL, La Liga, Serie A, Bundesliga, Ligue 1).
*   **Target Variables (Ground Truth):**
    *   Classification: `FTR` (Home, Draw, Away).
    *   Regression: `FTHG`, `FTAG` (Goals), `HS`, `AS`, `HST`, `AST` (Shots/Target), `HC`, `AC` (Corners), `HF`, `AF` (Fouls).

### 3.2 Engineered Features (Pre-Match Knowledge Base)
To prevent data leakage, the input tensor for Match $M_t$ will only contain aggregated data from Matches $M_{t-1}$ to $M_{t-k}$ (where $k$ is the sequence length, e.g., 5-10 matches).

1.  **Exponential Moving Averages (EMA):** Unlike simple rolling averages, EMA applies more weight to recent matches, accurately capturing current "form."
2.  **Relative Strength Index (Elo Approximation):** A dynamically updated rating for each team based on past opponents' strength (e.g., a win against Man City shifts the rating more than a win against a lower-table team).
3.  **Contextual Modifiers:**
    *   Rest Days: Days elapsed since the team's last competitive match (captures fatigue).
    *   Home/Away Flag: Boolean indicator to separate home advantage dynamics.

---

## 4. Deep Learning Architecture (Multi-Task LSTM with Attention)

The model will utilize a shared representation base, splitting into specialized "heads" for different prediction tasks.

### 4.1 Input Layer
*   **Shape:** `(Batch_Size, Sequence_Length, Number_of_Features)`
*   The input consists of two parallel sequences: The Home Team's last $k$ matches and the Away Team's last $k$ matches.

### 4.2 Temporal Extraction (Shared Layers)
1.  **Bidirectional LSTM (BiLSTM):** Captures temporal dependencies in both forward and backward contexts within the sequence window, understanding how a team's form is evolving (improving or declining).
2.  **Temporal Attention Mechanism:** Not all past matches are equally important. The attention layer allows the network to automatically assign higher weights to highly relevant past games (e.g., heavily weighting a recent game where a team generated 20 shots, rather than an anomalous game 4 weeks ago).
3.  **Feature Fusion:** The attention-weighted hidden states of the Home Team and Away Team are concatenated along with the *Static Context* (Rest days, current standings) into a unified dense representation vector.

### 4.3 Multi-Task Output Heads
The unified dense vector is passed through distinct, specialized neural network heads:

*   **Head 1: Match Outcome (Classification)**
    *   *Architecture:* Fully Connected Layer $\rightarrow$ Softmax.
    *   *Output:* `[P(Home), P(Draw), P(Away)]`
*   **Head 2: Match Goals (Regression)**
    *   *Architecture:* Fully Connected Layer $\rightarrow$ ReLU (to ensure non-negative outputs).
    *   *Output:* `[Expected_Home_Goals, Expected_Away_Goals]`
*   **Head 3: Offensive Statistics (Regression)**
    *   *Architecture:* Fully Connected Layer $\rightarrow$ ReLU.
    *   *Output:* `[Home_Shots, Away_Shots, Home_Corners, Away_Corners, ...]`

---

## 5. Training Strategy & Optimization

### 5.1 Multi-Task Loss Function
The network is optimized by minimizing a combined, weighted loss function:
$$L_{total} = \lambda_1 L_{outcome} + \lambda_2 L_{goals} + \lambda_3 L_{stats}$$
*   $L_{outcome}$: Cross-Entropy Loss (for predicting H/D/A).
*   $L_{goals}$: Poisson NLL Loss or Smooth L1 Loss (MSE) (Goals are count distributions).
*   $L_{stats}$: Mean Squared Error (MSE).
*   $\lambda_{1, 2, 3}$: Hyperparameters dictating how much the model should care about each task. (Tuning these prevents the model from obsessing over predicting corners at the expense of predicting the actual winner).

### 5.2 Transfer Learning Workflow
To build a highly robust model, training is split into two phases:
1.  **Phase 1: Universal Football Dynamics (Pre-Training).** The model is trained on a massive concatenated dataset of La Liga, Serie A, Bundesliga, and Ligue 1. The network learns universal rules (e.g., "teams that concede 15 shots a game tend to lose").
2.  **Phase 2: League Specialization (Fine-Tuning).** The base BiLSTM layers are frozen. The data is switched strictly to English Premier League data. Only the Multi-Task Output Heads are trained. This adapts the universal logic to the specific pacing and refereeing styles of the EPL.

---

## 6. Technology & Library Stack

*   **Data Processing:** `pandas`, `numpy`, `scikit-learn` (StandardScaler for normalizing shots/corners).
*   **Deep Learning Framework:** `PyTorch` (preferred for dynamic graph compilation and custom multi-task loss functions) or `PyTorch Lightning` (for boilerplate reduction).
*   **Hyperparameter Tuning:** `Optuna` (Automated searching for optimal learning rates, sequence lengths, and $\lambda$ loss weights).
*   **Experiment Tracking:** `MLflow` or `Weights & Biases (W&B)` to track which combinations of features yield the lowest validation loss.

---

## 7. Evaluation Metrics

A "damn good" predictor isn't evaluated just on accuracy, because predicting the favorite every time yields high accuracy but zero real-world value.

1.  **Log-Loss (Cross-Entropy):** Measures the confidence of the classification. (e.g., Being 99% confident in a wrong prediction is penalized heavily).
2.  **Brier Score:** Measures the accuracy of probabilistic predictions.
3.  **Macro F1-Score:** Ensures the model isn't just predicting "Home Win" every time; forces it to accurately identify Draws and Away upsets.
4.  **RMSE (Root Mean Square Error):** Used to evaluate the accuracy of the regression heads (Goals, Shots, Corners).

## 8. Deployment & Inference Flow
1. **State Hydration:** Look up the previous $k$ matches for all playing teams from the database to construct the $t-1$ input tensors.
2. **Forward Pass:** Pass tensors through the PyTorch model.
3. **Output Generation:** Output a JSON payload containing the probability matrix for H/D/A, Expected Goals (xG), and predicted match statistics to be consumed by a dashboard or application.