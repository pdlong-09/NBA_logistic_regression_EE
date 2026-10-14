# Predicting NBA games with logistic regression

> My IB Computer Science Extended Essay: a logistic regression model that predicts whether the home team wins an NBA game, using only what was known before tip-off. It was tested on the second half of the 2025–26 season and on the 2026 playoffs.

![Accuracy by pre-game Elo gap](fig_error_by_gap.png)

**Part of [phamduylong.com](https://phamduylong.com)** · `type: school` · `status: finished` · `Extended Essay, IB Computer Science`

---

## Why I built it

I play basketball and follow the NBA. For my Extended Essay I wanted to find out how far a simple, interpretable model can go at predicting games, and what it gets wrong, especially in the playoffs.

**Research question:** [Long: paste your exact research question here]

## Before → after

| Before | After |
| --- | --- |
| Always pick the home team: **56.8%** right on the test games | The full model: **72.3%** right on 523 unseen games (95% CI 68.5–75.9%) |

## Results

| Test | Games | Accuracy | Other |
| --- | --- | --- | --- |
| Hold-out (last 20% of games, Feb–Jun 2026) | 523 | **72.3%** | AUC 0.788 · Brier skill score 0.24 |
| Baseline: always pick the home team | 523 | 56.8% | |
| Baseline: Elo rating only | 523 | 72.1% | |
| 2026 playoffs only (18 Apr – 13 Jun 2026) | 83 | **65.1%** | Brier 0.217 · log-loss 0.624 |
| Time-series cross-validation, 5 folds | — | 65.6% ± 4.5 | AUC 0.715 ± 0.047 |

What the numbers say:

- **Elo does almost all the work.** Adding eight more features on top of Elo moved accuracy from 72.1% to 72.3%. A McNemar test finds no real difference (p = 1.0). The lasso confirms this: only Elo and 10-game net rating carry large weights, and two features are dropped to zero.
- **Playoffs are harder.** In the hold-out set the model was right on 74.2% of regular-season games but 68.1% of playoff games (Fisher p = 0.17, so a direction rather than proof). On the 2026 playoffs alone it was right 65.1% of the time.
- **It knows when it's guessing.** When the model is more than 70% confident it's right 85.6% of the time. On toss-ups, where the Elo gap is under 50, it's barely better than a coin (58.6%).
- **A more complex model didn't help.** Gradient-boosted trees scored lower than logistic regression on both the hold-out split and cross-validation.

## Features

Every feature is a **home-minus-away difference**, computed only from games played *before* the one being predicted.

| Feature | Meaning |
| --- | --- |
| `diff_elo` | Elo rating before the game (start 1500, K = 20, +100 home-court in the expectation) |
| `diff_net10` | Net rating (points scored minus allowed per 100 possessions), average of the last 10 games |
| `diff_net3` | Net rating, last 3 games |
| `diff_ts10` | True shooting %, last 10 games |
| `diff_tov10` | Turnover %, last 10 games |
| `diff_ftr10` | Free-throw rate, last 10 games |
| `diff_ast10` | Assists, last 10 games |
| `diff_restdays` | Days of rest since the previous game |
| `diff_distance_km` | Distance travelled from the previous game's arena (haversine) |

## How it works

1. **`building.py`** turns raw team box scores into one row per game.
   - It reads home/away and the opponent from the match-up text.
   - It computes possessions and the efficiency stats.
   - It builds the rolling averages with a one-game shift, so a game never sees its own result.
   - It runs Elo through the season in date order and works out rest days and travel.
   - It then trains the model on the first 80% of games in time order and tests on the last 20%. There's no shuffling, so the model never learns from the future.
2. **`evaluation.py`** re-fits the model and runs the checks behind the Results section:
   - confusion matrix and baselines
   - Brier and log-loss, McNemar and bootstrap tests, and multicollinearity (VIF)
   - feature-group ablation and error analysis by Elo gap and by confidence
   - regular season vs playoffs, time-series cross-validation, lasso, and a gradient-boosted comparison
   - all seven figures
3. **`experiment.py`** is the playoff test. It trains only on games before 18 April 2026, then predicts all 83 playoff games.

## Built with

`Python` · `pandas` · `NumPy` · `scikit-learn` · `SciPy` · `Matplotlib`

## Run it

```bash
git clone https://github.com/pdlong-09/NBA_logistic_regression_EE.git
cd NBA_logistic_regression_EE
pip install -r requirements.txt
python evaluation.py     # rebuilds evaluation_report.csv and all figures
python experiment.py     # re-runs the 2026 playoff test
```

`evaluation.py` and `experiment.py` run from the included `nba_game_table.csv`. `building.py` needs the raw files `team_box.csv` and `team_arena.csv`, which aren't in this repo. Source: [Long: where the box scores came from].

## Files

| File | What it is |
| --- | --- |
| `building.py` | Feature engineering, Elo, game table, first model |
| `evaluation.py` | Full evaluation and figures |
| `experiment.py` | 2026 playoff test |
| `nba_game_table.csv` | 2,615 games, 24 Oct 2024 – 13 Jun 2026, with all features |
| `experiment_predictions.csv` | Model predictions for the 83 playoff games |
| `nba_model_metrics.csv`, `nba_model_coefficients.csv` | Hold-out metrics and coefficients |
| `evaluation_report.csv` | Full text report from `evaluation.py` |
| `experiment_metrics.csv` | Playoff test metrics |
| `fig_*.png` | Confusion matrix, ROC, calibration, coefficients, ablation, error by Elo gap, model comparison |

## What I learned

[Long: 2–3 sentences in your own words. For example, why Elo alone was so hard to beat, or what surprised you about the playoffs.]

## Limitations

- **Only about two seasons of data** (2,615 games), and only at team level. There's nothing on injuries, rotations or who actually played, which matters most in the playoffs.
- **Elo starts every team at 1,500 in October 2024**, so early-season ratings carry little information.
- **The hold-out window mixes regular-season and playoff games.** That's why the playoffs also get their own separate test.
- **The raw box-score files aren't included**, so `building.py` can't be re-run from this repo alone.

## Figures

<p>
  <img src="fig_ablation.png" width="49%" alt="Feature-group ablation">
  <img src="fig_roc.png" width="49%" alt="ROC curve">
  <img src="fig_calibration.png" width="49%" alt="Calibration">
  <img src="fig_coefficients.png" width="49%" alt="Coefficients">
  <img src="fig_confusion_matrix.png" width="49%" alt="Confusion matrix">
  <img src="fig_model_compare.png" width="49%" alt="Logistic vs gradient-boosted trees">
</p>

---

<sub>Phạm Duy Long · Hà Nội · [phamduylong.com](https://phamduylong.com)</sub>
