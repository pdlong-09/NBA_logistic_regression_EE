import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import binomtest, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
FEATURES = ["diff_elo", "diff_net10", "diff_net3", "diff_ts10", "diff_tov10", "diff_ftr10", "diff_ast10", "diff_restdays", "diff_distance_km"]

#setup and fit model
games = pd.read_csv("nba_game_table.csv")
games["Game Date"] = pd.to_datetime(games["Game Date"])
games = games.sort_values(["Game Date", "GAME_ID"]).reset_index(drop=True)
split = int(len(games) * 0.8)
tr, te = games.iloc[:split], games.iloc[split:]
ytr, yte = tr["home_win"].values, te["home_win"].values
base_rate = ytr.mean()
brier_ref = brier_score_loss(yte, np.full_like(yte, base_rate, dtype=float))
lines = []
def log(s): print(s); lines.append(s)
def fit_logreg(cols):
    sc = StandardScaler(); Xt = sc.fit_transform(tr[cols]); Xe = sc.transform(te[cols])
    m = LogisticRegression(max_iter=2000).fit(Xt, ytr)
    p = m.predict(Xe); pr = m.predict_proba(Xe)[:, 1]
    return (accuracy_score(yte, p), roc_auc_score(yte, pr),
            1 - brier_score_loss(yte, pr) / brier_ref, p, pr, m, sc)
acc_full, auc_full, bss_full, pred, prob, model, scaler = fit_logreg(FEATURES)

#confusion matrix
cm = confusion_matrix(yte, pred)
log("CONFUSION MATRIX")
log(f"            pred_away  pred_home")
log(f"act_away    {cm[0,0]:8d}   {cm[0,1]:8d}")
log(f"act_home    {cm[1,0]:8d}   {cm[1,1]:8d}")
log(f"accuracy={acc_full:.4f}  precision={precision_score(yte,pred):.4f}"
    f"  recall={recall_score(yte,pred):.4f}  f1={f1_score(yte,pred):.4f}")

#baselines
always_home = np.ones_like(yte)
majority = np.full_like(yte, int(round(ytr.mean())))
acc_elo, auc_elo, bss_elo, elo_pred, elo_prob, _, _ = fit_logreg(["diff_elo"])
log("\nBASELINES")
log(f"always-home     : {accuracy_score(yte, always_home):.4f}")
log(f"majority-class  : {accuracy_score(yte, majority):.4f}")
log(f"elo-only        : {acc_elo:.4f}")
log(f"full model      : {acc_full:.4f}")

#Brier score
brier = brier_score_loss(yte, prob)
log("\nBRIER SCORES")
log(f"Brier score          : {brier:.4f}")
log(f"Brier (base-rate ref): {brier_ref:.4f}")
log(f"Brier Skill Score    : {1 - brier/brier_ref:.4f}   (>0 beats the base-rate forecast)")
log(f"log-loss             : {log_loss(yte, prob):.4f}")

#Statistical interference
model_correct = (pred == yte); ah_correct = (always_home == yte)
b = int(np.sum(model_correct & ~ah_correct))   #model right only
c = int(np.sum(~model_correct & ah_correct))   #always-home right only
p_mcnemar = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
log("\nSTATISTICAL INFERENCE")
log(f"McNemar vs always-home: b(model-only right)={b}, c(always-home-only right)={c}, exact p={p_mcnemar:.4g}")
log(f"  -> {'model differs significantly from always-home' if p_mcnemar<0.05 else 'not significant'}")

rng = np.random.default_rng(0)
accs = [accuracy_score(yte[idx], pred[idx]) for idx in
        (rng.integers(0, len(yte), len(yte)) for _ in range(5000))]
lo, hi = np.percentile(accs, [2.5, 97.5])
log(f"Bootstrap 95% CI for accuracy: [{lo:.4f}, {hi:.4f}]  (point {acc_full:.4f})")

#VIF
X = games[FEATURES].fillna(0).values; X = (X - X.mean(0)) / X.std(0)
vif = np.diag(inv(np.corrcoef(X, rowvar=False)))
log("\nMULTICOLLINEARITY (VIF)")
for f, v in sorted(zip(FEATURES, vif), key=lambda t: -t[1]): log(f"  {v:5.2f}  {f}")

#feature group test
log("\nINCREMENTAL FEATURE-GROUP ABLATION")
groups = [
    ("Elo only",             ["diff_elo"]),
    ("+ recent form",        ["diff_elo", "diff_net10", "diff_net3"]),
    ("+ style",              ["diff_elo", "diff_net10", "diff_net3", "diff_ts10", "diff_tov10", "diff_ftr10", "diff_ast10"]),
    ("+ situational (full)", FEATURES),
]
abl = {}
for name, cols in groups:
    if cols == FEATURES:                
        a, au, bss, p = acc_full, auc_full, bss_full, pred
    elif cols == ["diff_elo"]:         
        a, au, bss, p = acc_elo, auc_elo, bss_elo, elo_pred
    else:
        a, au, bss, p, _, _, _ = fit_logreg(cols)
    abl[name] = dict(acc=a, auc=au, bss=bss, pred=p)
    log(f"  {name:22s} k={len(cols):2d}  acc={a:.4f}  AUC={au:.4f}  BSS={bss:.4f}")

#full model vs elo only
b2 = int(((pred == yte) & (elo_pred != yte)).sum())   #full right only
c2 = int(((pred != yte) & (elo_pred == yte)).sum())   #elo right only
p_fe = binomtest(min(b2, c2), b2 + c2, 0.5).pvalue if (b2 + c2) else 1.0
log("\nMcNemar: FULL vs ELO-ONLY")
log(f"  b(full-only right)={b2}, c(elo-only right)={c2}, exact p={p_fe:.3f} -> "
    f"{'significant' if p_fe < 0.05 else 'NOT significant: the 8 extra features add no reliable gain'}")

#error analysis
te2 = te.copy(); te2["correct"] = (pred == yte); te2["gap"] = te2["diff_elo"].abs()
te2["conf"] = np.maximum(prob, 1 - prob)
log("\nERROR ANALYSIS: accuracy by pre-game |Elo gap|")
gap_bins = [(0, 50, "toss-up <50"), (50, 150, "moderate 50-150"), (150, 1e9, "lopsided >150")]
gap_res = []
for glo, ghi, lab in gap_bins:
    m = (te2["gap"] >= glo) & (te2["gap"] < ghi)
    if m.sum():
        acc = te2["correct"][m].mean(); gap_res.append((lab, int(m.sum()), acc))
        log(f"  {lab:16s} n={m.sum():3d}  acc={acc:.4f}")
log("  accuracy by model confidence:")
for clo, chi in [(0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]:
    m = (te2["conf"] >= clo) & (te2["conf"] < chi)
    if m.sum(): log(f"    conf {clo:.2f}-{chi:.2f}  n={m.sum():3d}  acc={te2['correct'][m].mean():.4f}")

#regular season vs playoffs
pw = te2["Game Date"].dt.month.between(4, 6)
rc, rn = te2["correct"][~pw].sum(), (~pw).sum()
oc, on = te2["correct"][pw].sum(), pw.sum()
_, p_fish = fisher_exact([[rc, rn - rc], [oc, on - oc]])
log("\nREGULAR vs PLAYOFF")
log(f"  regular {rc}/{rn}={rc/rn:.3f} | playoff {oc}/{on}={oc/on:.3f} | Fisher exact p={p_fish:.3f} "
    f"({'significant' if p_fish < 0.05 else 'directional, not significant alone'})")

#time-series cross validation
log("\nTIME-SERIES CROSS-VALIDATION (5 expanding folds, logistic)")
tss = TimeSeriesSplit(n_splits=5)
Xall, yall = games[FEATURES].values, games["home_win"].values
lr_acc, lr_auc = [], []
for tri, tei in tss.split(Xall):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xall[tri], yall[tri])
    pp = pipe.predict_proba(Xall[tei])[:, 1]
    lr_acc.append(accuracy_score(yall[tei], (pp >= .5).astype(int))); lr_auc.append(roc_auc_score(yall[tei], pp))
log(f"  acc per fold: {[round(x,3) for x in lr_acc]}  mean={np.mean(lr_acc):.3f} sd={np.std(lr_acc):.3f}")
log(f"  AUC per fold: {[round(x,3) for x in lr_auc]}  mean={np.mean(lr_auc):.3f} sd={np.std(lr_auc):.3f}")

#lasso check
log("\nL1 (lasso) COEFFICIENTS")
sc1 = StandardScaler(); Xt1 = sc1.fit_transform(tr[FEATURES])
l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=5000).fit(Xt1, ytr)
for f, co in sorted(zip(FEATURES, l1.coef_[0]), key=lambda t: -abs(t[1])):
    log(f"  {co:+.3f}  {f}" + ("   <- zeroed" if abs(co) < 1e-6 else ""))

#compare another model
log("\nGRADIENT-BOOSTED TREES vs LOGISTIC REGRESSION")
gbt = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                     max_iter=300, l2_regularization=1.0,
                                     early_stopping=False, random_state=0)
gbt.fit(tr[FEATURES].values, ytr)
gp = gbt.predict(te[FEATURES].values); gpr = gbt.predict_proba(te[FEATURES].values)[:, 1]
log("  single hold-out split:")
log(f"    logistic : acc={acc_full:.4f}  AUC={auc_full:.4f}  BSS={bss_full:.4f}")
log(f"    GBT      : acc={accuracy_score(yte,gp):.4f}  AUC={roc_auc_score(yte,gpr):.4f}  "
    f"BSS={1-brier_score_loss(yte,gpr)/brier_ref:.4f}")
gbt_acc, gbt_auc = [], []
for tri, tei in tss.split(Xall):
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                       l2_regularization=1.0, early_stopping=False, random_state=0)
    m.fit(Xall[tri], yall[tri]); pp = m.predict_proba(Xall[tei])[:, 1]
    gbt_acc.append(accuracy_score(yall[tei], (pp >= .5).astype(int))); gbt_auc.append(roc_auc_score(yall[tei], pp))
log("  time-series CV (5 folds):")
log(f"    logistic : acc {np.mean(lr_acc):.3f}+/-{np.std(lr_acc):.3f}   AUC {np.mean(lr_auc):.3f}+/-{np.std(lr_auc):.3f}")
log(f"    GBT      : acc {np.mean(gbt_acc):.3f}+/-{np.std(gbt_acc):.3f}   AUC {np.mean(gbt_auc):.3f}+/-{np.std(gbt_auc):.3f}")

pd.DataFrame({"line": lines}).to_csv("evaluation_report.csv", index=False)

#confusion figure
fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["Away win", "Home win"]); ax.set_yticks([0, 1], ["Away win", "Home win"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (hold-out)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
fig.tight_layout(); fig.savefig("fig_confusion_matrix.png"); plt.close(fig)

#calibration figure
bins = np.linspace(0, 1, 11); idx = np.digitize(prob, bins) - 1
xs, ys = [], []
for bck in range(10):
    m = idx == bck
    if m.sum() >= 5: xs.append(prob[m].mean()); ys.append(yte[m].mean())
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
ax.plot(xs, ys, "o-", color="#1f4e79", label="model")
ax.set_xlabel("Predicted P(home win)"); ax.set_ylabel("Observed frequency")
ax.set_title("Calibration curve"); ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig("fig_calibration.png"); plt.close(fig)

#coefficient figure
coef = pd.DataFrame({"feature": FEATURES, "coef": model.coef_[0]}).sort_values("coef")
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#c0392b" if cc < 0 else "#1f4e79" for cc in coef["coef"]]
ax.barh(coef["feature"], coef["coef"], color=colors)
ax.axvline(0, color="black", lw=0.8); ax.set_xlabel("Standardized logistic coefficient")
ax.set_title("Feature coefficients (log-odds, standardized)")
fig.tight_layout(); fig.savefig("fig_coefficients.png"); plt.close(fig)

#roc figure
fpr, tpr, _ = roc_curve(yte, prob)
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot(fpr, tpr, color="#1f4e79", label=f"logistic (AUC={auc_full:.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", label="random (0.5)")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curve (hold-out)"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("fig_roc.png"); plt.close(fig)

#ablation figure
fig, ax = plt.subplots(figsize=(5.6, 3.8))
xs = np.arange(len(groups))
ax.plot(xs, [abl[n]["acc"] for n, _ in groups], "o-", color="#1f4e79", label="Accuracy")
ax.plot(xs, [abl[n]["auc"] for n, _ in groups], "s--", color="#c0392b", label="AUC")
ax.axhline((yte == 1).mean(), color="grey", ls=":", label=f"always-home ({(yte==1).mean():.3f})")
ax.set_xticks(xs, [n.replace(" ", "\n", 1) for n, _ in groups]); ax.set_ylim(0.5, 0.85)
ax.set_ylabel("Hold-out score"); ax.set_title("Incremental feature-group ablation"); ax.legend()
fig.tight_layout(); fig.savefig("fig_ablation.png"); plt.close(fig)

#elo gap figure
fig, ax = plt.subplots(figsize=(5.0, 3.8))
labs = [r[0].split()[0] + "\n" + r[0].split(maxsplit=1)[1] for r in gap_res]
vals = [r[2] for r in gap_res]; ns = [r[1] for r in gap_res]
bars = ax.bar(labs, vals, color=["#c0392b", "#e0a458", "#1f4e79"])
ax.axhline(0.5, color="grey", ls=":"); ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy"); ax.set_title("Accuracy by pre-game Elo gap")
for bb, n in zip(bars, ns): ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.02, f"n={n}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig("fig_error_by_gap.png"); plt.close(fig)

#model comparison figure
fig, ax = plt.subplots(figsize=(5.2, 3.8))
xpos = np.arange(2)
means = [np.mean(lr_auc), np.mean(gbt_auc)]; sds = [np.std(lr_auc), np.std(gbt_auc)]
ax.bar(xpos, means, yerr=sds, capsize=6, color=["#1f4e79", "#2e8b57"])
ax.set_xticks(xpos, ["Logistic\nregression", "Gradient-boosted\ntrees"])
ax.axhline(0.5, color="grey", ls=":", label="random")
ax.set_ylim(0.5, 0.85); ax.set_ylabel("AUC (5-fold time-series CV)")
ax.set_title("Commparing to non-linear model")
for i, (mn, sd) in enumerate(zip(means, sds)): ax.text(i, mn + sd + 0.01, f"{mn:.3f}", ha="center", fontsize=10)
fig.tight_layout(); fig.savefig("fig_model_compare.png"); plt.close(fig)

print("\nSaved: evaluation_report.csv + 7 figures")
