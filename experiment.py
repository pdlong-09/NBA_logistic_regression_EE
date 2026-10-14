import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["diff_elo", "diff_net10", "diff_net3", "diff_ts10", "diff_tov10", "diff_ftr10", "diff_ast10", "diff_restdays", "diff_distance_km"]
games = pd.read_csv("nba_game_table.csv")
games["Game Date"] = pd.to_datetime(games["Game Date"])

#log game
exp = pd.read_csv("experiment_predictions.csv")
exp["Game Date"] = pd.to_datetime(exp["Game Date"])
window_start = exp["Game Date"].min()
key_cols = ["Game Date", "home_team", "away_team"]
merged = exp[key_cols].merge(games, on=key_cols, how="left")
missing = merged["home_win"].isna().sum()
if missing:
  print(f"WARNING: {missing} logged games not found in corrected table")
  merged = merged.dropna(subset=["home_win"])

#train
train = games[games["Game Date"] < window_start].copy()
scaler = StandardScaler()
Xtr = scaler.fit_transform(train[FEATURES])
model = LogisticRegression(max_iter=2000)
model.fit(Xtr, train["home_win"].values)
Xexp = scaler.transform(merged[FEATURES])

#predict
merged["predicted_home_win"] = model.predict(Xexp)
merged["predicted_probability_home_win"] = model.predict_proba(Xexp)[:, 1]
merged["home_win"] = merged["home_win"].astype(int)
out = merged[["Game Date", "home_team", "away_team", "home_win", "predicted_home_win", "predicted_probability_home_win"]]
out.to_csv("experiment_predictions_reproduced.csv", index=False)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, log_loss
y, p, pr = out["home_win"], out["predicted_home_win"], out["predicted_probability_home_win"]
pd.DataFrame({
    "metric": ["accuracy", "precision", "recall", "f1", "brier", "log_loss", "rows"],
    "value": [accuracy_score(y, p), precision_score(y, p, zero_division=0), recall_score(y, p, zero_division=0), f1_score(y, p, zero_division=0), brier_score_loss(y, pr), log_loss(y, pr), len(out)]
}).to_csv("experiment_metrics.csv", index=False)
print(f"Trained on {len(train)} pre-window games; tested on {len(out)} playoff games")
print(f"acc={accuracy_score(y,p)}  prec={precision_score(y,p)}  rec={recall_score(y,p)}"
      f"f1={f1_score(y,p)}  brier={brier_score_loss(y,pr)}")
