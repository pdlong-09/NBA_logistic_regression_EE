import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#helper functions
def parse_matchup(s):
    if "vs." in s:
        a, b = s.split(" vs. ")
        return a.strip(), b.strip(), 1
    a, b = s.split(" @ ")
    return a.strip(), b.strip(), 0
def rolling_prev(series, window):
    return series.shift(1).rolling(window, min_periods=1).mean()
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat1 - lat2) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon1 - lon2) / 2) ** 2
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

#load data
team = pd.read_csv("team_box.csv")
arenas = pd.read_csv("team_arena.csv").rename(columns={team.columns[0]: "Team"})
arenas = pd.read_csv("team_arena.csv")
arenas.columns = ["Team", "arena", "lat", "long"]
team.columns = [c.replace("\ufeff", "") for c in team.columns]
team["Game Date"] = pd.to_datetime(team["Game Date"], format="%d/%m/%y")
team = team.drop_duplicates().dropna().copy()

#get opponent and home/away from matchup
parsed = team["Match Up"].apply(lambda s: pd.Series(parse_matchup(s), index=["Team_", "Opponent", "Home"]))
team["Opponent"] = parsed["Opponent"]
team["Home"] = parsed["Home"].astype(int)
def game_key(row):
    a, b = sorted([row["Team"], row["Opponent"]])
    return f"G_{row['Game Date'].date()}_{a}_{b}"
team["GAME_ID"] = team.apply(game_key, axis=1)
opp_pts = team[["GAME_ID", "Team", "PTS"]].rename(columns={"Team": "Opponent", "PTS": "Opp_PTS"})
team = team.merge(opp_pts, on=["GAME_ID", "Opponent"], how="left")

#keep complete games only
counts = team.groupby("GAME_ID")["Team"].transform("count")
team = team[counts == 2].copy()
team = team.dropna(subset=["Opp_PTS"])
team["Opp_PTS"] = team["Opp_PTS"].astype(int)
team["Win"] = (team["W/L"] == "W").astype(int)

#make box score stats
team["Poss"] = team["FGA"] - team["OREB"] + team["TOV"] + 0.44 * team["FTA"]
team = team[(team["Poss"] > 0) & (team["FGA"] > 0)].copy()
team["ORtg"]    = 100 * team["PTS"] / team["Poss"]#offensive rating: offense per 100 possessions
team["DRtg"]    = 100 * team["Opp_PTS"] / team["Poss"]#defensive rating: defense per 100 possessions
team["NetRtg"]  = team["ORtg"] - team["DRtg"]#net rating = offensive rating-defensive rating
team["TS_pct"]  = team["PTS"] / (2 * (team["FGA"] + 0.44 * team["FTA"]))#shooting percentage
team["TOV_pct"] = team["TOV"] / team["Poss"]#turnover percentage
team["FTr"]     = team["FTA"] / team["FGA"]#free throw rate
team["AST_g"]   = team["AST"]#assist

#recent form
team = team.sort_values(["Team", "Game Date", "GAME_ID"]).reset_index(drop=True)
roll10 = {"net10": "NetRtg", "ts10": "TS_pct", "tov10": "TOV_pct", "ftr10": "FTr", "ast10": "AST_g"}
roll3  = {"net3": "NetRtg"}
for new, src in roll10.items():
    team[new] = team.groupby("Team")[src].apply(lambda s: rolling_prev(s, 10)).reset_index(level=0, drop=True)
for new, src in roll3.items():
    team[new] = team.groupby("Team")[src].apply(lambda s: rolling_prev(s, 3)).reset_index(level=0, drop=True)

#rest days and travel distance
team["prev_date"] = team.groupby("Team")["Game Date"].shift(1)
team["restdays"] = ((team["Game Date"] - team["prev_date"]).dt.days - 1).fillna(0).clip(lower=0)
coord = arenas.set_index("Team")[["lat", "long"]].to_dict("index")
def venue_of(row):
    host = row["Team"] if row["Home"] == 1 else row["Opponent"]
    c = coord.get(host)
    return (c["lat"], c["long"]) if c else (np.nan, np.nan)
venues = team.apply(venue_of, axis=1, result_type="expand")
team["venue_lat"], team["venue_long"] = venues[0], venues[1]
team["prev_venue_lat"] = team.groupby("Team")["venue_lat"].shift(1)
team["prev_venue_long"] = team.groupby("Team")["venue_long"].shift(1)
def travel(row):
    if pd.isna(row["prev_venue_lat"]):
        return 0.0
    return haversine_km(row["prev_venue_lat"], row["prev_venue_long"], row["venue_lat"], row["venue_long"])
team["distance_km"] = team.apply(travel, axis=1)

#elo rating
team = team.sort_values(["Game Date", "GAME_ID", "Home"], ascending=[True, True, False]).reset_index(drop=True)
elo = {t: 1500.0 for t in team["Team"].unique()}
team["Elo"] = np.nan
for gid, g in team.groupby("GAME_ID", sort=False):
    if len(g) != 2:
        continue
    h = g[g["Home"] == 1]; a = g[g["Home"] == 0]
    if len(h) != 1 or len(a) != 1:
        continue
    hi, ai = h.index[0], a.index[0]
    ht, at = h["Team"].iloc[0], a["Team"].iloc[0]
    team.loc[hi, "Elo"] = elo[ht]           #pre-game elo
    team.loc[ai, "Elo"] = elo[at]
    E_home = 1 / (1 + 10 ** (-(elo[ht] - elo[at] + 100) / 400))
    S_home = h["Win"].iloc[0]
    elo[ht] += 20 * (S_home - E_home)
    elo[at] += 20 * ((1 - S_home) - (1 - E_home))

team = team.dropna(subset=list(roll10) + list(roll3) + ["Elo"])
team = team.sort_values(["Game Date", "GAME_ID", "Home"], ascending=[True, True, False]).reset_index(drop=True)
team.to_csv("nba_teams_table.csv", index=False)

#build game table
keep = list(roll10) + list(roll3) + ["restdays", "distance_km", "Elo"]
home = team[team["Home"] == 1].copy()
away = team[team["Home"] == 0].copy()
h = home[["GAME_ID", "Game Date", "Team", "Opponent", "Win"] + keep].rename(
    columns={"Team": "home_team", "Opponent": "away_team", "Win": "home_win", **{k: "home_" + k for k in keep}})
a = away[["GAME_ID"] + keep].rename(columns={k: "away_" + k for k in keep})
games = h.merge(a, on="GAME_ID", how="inner").sort_values("Game Date").reset_index(drop=True)
FEATURES = ["diff_elo", "diff_net10", "diff_net3", "diff_ts10", "diff_tov10","diff_ftr10", "diff_ast10", "diff_restdays", "diff_distance_km"]
games["diff_elo"]         = games["home_Elo"]      - games["away_Elo"]
games["diff_net10"]       = games["home_net10"]    - games["away_net10"]
games["diff_net3"]        = games["home_net3"]     - games["away_net3"]
games["diff_ts10"]        = games["home_ts10"]     - games["away_ts10"]
games["diff_tov10"]       = games["home_tov10"]    - games["away_tov10"]
games["diff_ftr10"]       = games["home_ftr10"]    - games["away_ftr10"]
games["diff_ast10"]       = games["home_ast10"]    - games["away_ast10"]
games["diff_restdays"]    = games["home_restdays"] - games["away_restdays"]
games["diff_distance_km"] = games["home_distance_km"] - games["away_distance_km"]
games.to_csv("nba_game_table.csv", index=False)

train_table = games[["GAME_ID", "Game Date", "home_team", "away_team", "home_win"] + FEATURES].copy()
train_table.to_csv("nba_training_table.csv", index=False)

#train and test model
train_table = train_table.sort_values(["Game Date", "GAME_ID"]).reset_index(drop=True)
split = int(len(train_table) * 0.8)
tr, te = train_table.iloc[:split], train_table.iloc[split:]
scaler = StandardScaler()
Xtr = scaler.fit_transform(tr[FEATURES]); Xte = scaler.transform(te[FEATURES])
ytr, yte = tr["home_win"].values, te["home_win"].values
model = LogisticRegression(max_iter=2000)
model.fit(Xtr, ytr)
pred = model.predict(Xte)
prob = model.predict_proba(Xte)[:, 1]
pd.DataFrame({
    "metric": ["accuracy", "precision", "recall", "f1", "train_rows", "test_rows"],
    "value": [accuracy_score(yte, pred), precision_score(yte, pred, zero_division=0),
              recall_score(yte, pred, zero_division=0), f1_score(yte, pred, zero_division=0),
              len(tr), len(te)]
}).to_csv("nba_model_metrics.csv", index=False)
pd.DataFrame({"feature": FEATURES, "coefficient": model.coef_[0]}) \
    .sort_values("coefficient", ascending=False).to_csv("nba_model_coefficients.csv", index=False)
out = te[["GAME_ID", "Game Date", "home_team", "away_team", "home_win"]].copy()
out["predicted_home_win"] = pred
out["predicted_probability_home_win"] = prob
out.to_csv("nba_test_predictions.csv", index=False)

print(f"Games built: {len(games)}   train: {len(tr)}   test: {len(te)}")
print(f"Home win rate (overall): {games['home_win'].mean():.3f}")
print(f"Hold-out  acc={accuracy_score(yte,pred):.3f}  prec={precision_score(yte,pred):.3f}"
      f"  rec={recall_score(yte,pred):.3f}  f1={f1_score(yte,pred):.3f}")
