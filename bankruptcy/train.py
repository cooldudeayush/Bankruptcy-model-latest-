"""Train one LightGBM model per prediction horizon (1-5 years ahead).

    python -m bankruptcy.train                # full run: feature selection + tuning + evaluation
    python -m bankruptcy.train --trials 5     # quick run

Pipeline for each horizon
  1. Load the matching .arff file and split 85% train / 15% test (stratified, same rows as v1).
  2. Pick the most useful ratios using cross-validation on the training part only.
  3. Tune LightGBM hyper-parameters with Optuna (5-fold CV, early stopping on average precision).
  4. Get out-of-fold predictions and use them to
       - calibrate scores into real probabilities (Platt scaling), and
       - choose the decision threshold that maximises F1.
  5. Fit the final model on the whole training part and score the untouched test part once.
Outputs: models/model_{h}y.txt, models/metadata.json, reports/metrics.json, reports/figures/*.png
"""
import argparse
import json
import platform
from datetime import date

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import (CV_FOLDS, HORIZONS, MODELS_DIR, N_FEATURES, N_TRIALS, RANDOM_STATE,
                     REPORTS_DIR, file_for_horizon)
from .data import load_horizon, train_test
from .features import RATIO_COLUMNS

BASE_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "bagging_freq": 1,
    "feature_pre_filter": False,
    "deterministic": True,
    "force_col_wise": True,
    "seed": RANDOM_STATE,
    "num_threads": 4,
    "verbosity": -1,
}
DEFAULT_PARAMS = {
    "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 20,
    "feature_fraction": 0.7, "bagging_fraction": 0.8, "lambda_l2": 1.0,
    "pos_weight": "balanced",
}
MAX_ROUNDS = 5000
EARLY_STOPPING = 200
OOF_REPEATS = 3


# ----------------------------------------------------------------------------- helpers
def _lgb_params(params: dict, y: pd.Series) -> dict:
    p = {**BASE_PARAMS, **params}
    ratio = (len(y) - y.sum()) / max(y.sum(), 1)
    mode = p.pop("pos_weight", "none")
    p["scale_pos_weight"] = {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "balanced": float(ratio)}[mode]
    return p


def fit(params, X_a, y_a, X_b=None, y_b=None, rounds=MAX_ROUNDS) -> lgb.Booster:
    p = _lgb_params(params, y_a)
    train = lgb.Dataset(X_a, y_a, free_raw_data=False)
    if X_b is None:
        return lgb.train(p, train, num_boost_round=rounds)
    valid = lgb.Dataset(X_b, y_b, reference=train)
    return lgb.train(p, train, num_boost_round=rounds, valid_sets=[valid],
                     callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])


def cross_validate(params, X, y, repeats=1, seed=RANDOM_STATE):
    """Out-of-fold raw scores (repeats x n_rows), best iterations and gain importances."""
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=repeats, random_state=seed)
    oof = np.zeros((repeats, len(y)))
    iters, gains = [], []
    for i, (a, b) in enumerate(cv.split(X, y)):
        booster = fit(params, X.iloc[a], y.iloc[a], X.iloc[b], y.iloc[b])
        oof[i // CV_FOLDS, b] = booster.predict(X.iloc[b], raw_score=True,
                                                num_iteration=booster.best_iteration)
        iters.append(booster.best_iteration)
        gains.append(booster.feature_importance("gain"))
    return oof, iters, np.mean(gains, axis=0)


def mean_ap(y, oof):
    return float(np.mean([average_precision_score(y, row) for row in oof]))


def select_features(X, y, k=N_FEATURES):
    """Rank all 64 ratios by average LightGBM gain across CV folds (training data only)."""
    _, _, gain = cross_validate(DEFAULT_PARAMS, X, y)
    ranking = pd.Series(gain, index=X.columns).sort_values(ascending=False)
    return list(ranking.index[:k]), ranking


def tune(X, y, n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "pos_weight": trial.suggest_categorical("pos_weight", ["none", "sqrt", "balanced"]),
        }
        oof, _, _ = cross_validate(params, X, y, seed=RANDOM_STATE + 1)
        return mean_ap(y, oof)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.enqueue_trial(DEFAULT_PARAMS)          # the defaults are always one of the candidates
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def calibrate(raw, y):
    """Platt scaling: probability = sigmoid(a * raw_score + b)."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(raw.reshape(-1, 1), y)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def to_prob(raw, cal):
    return 1.0 / (1.0 + np.exp(-(cal["a"] * raw + cal["b"])))


def best_threshold(y, p):
    grid = np.round(np.arange(0.02, 0.96, 0.01), 2)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "accuracy": (tp + tn) / len(y),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ci(y, p, thr, n=1000, seed=RANDOM_STATE) -> dict:
    """95% intervals for the test metrics: the test set has only ~40-80 bankruptcies."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    out = {"f1": [], "roc_auc": [], "pr_auc": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        out["f1"].append(f1_score(y[i], (p[i] >= thr).astype(int), zero_division=0))
        out["roc_auc"].append(roc_auc_score(y[i], p[i]))
        out["pr_auc"].append(average_precision_score(y[i], p[i]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def pick_examples(X_te, y_te, p_te, thr):
    """Three real (unseen) test companies with complete data, to try in the app."""
    df = X_te.assign(_y=y_te.values, _p=p_te)
    df = df[df[list(X_te.columns)].notna().all(axis=1)]
    healthy = df[df._y == 0].sort_values("_p")
    safe = healthy.iloc[len(healthy) // 2]
    risky = df[(df._y == 1) & (df._p >= thr)].sort_values("_p", ascending=False).iloc[0]
    border = df.iloc[(df._p - 0.75 * thr).abs().argsort().iloc[0]]   # inside the "Medium" band
    clean = lambda r: {k: (None if pd.isna(v) else float(v)) for k, v in r.drop(["_y", "_p"]).items()}
    return {
        "Healthy company": {"values": clean(safe), "actually_bankrupt": bool(safe._y)},
        "Borderline company": {"values": clean(border), "actually_bankrupt": bool(border._y)},
        "Distressed company": {"values": clean(risky), "actually_bankrupt": bool(risky._y)},
    }


# ----------------------------------------------------------------------------- per horizon
def train_horizon(h: int, n_trials: int) -> dict:
    df = load_horizon(h)
    X_tr, X_te, y_tr, y_te = train_test(df)

    feats, ranking = select_features(X_tr, y_tr)
    X_tr, X_te = X_tr[feats], X_te[feats]

    params, cv_ap_tuning = tune(X_tr, y_tr, n_trials)

    oof, iters, _ = cross_validate(params, X_tr, y_tr, repeats=OOF_REPEATS, seed=RANDOM_STATE + 2)
    cal = calibrate(oof.ravel(), np.tile(y_tr.values, OOF_REPEATS))
    oof_p = to_prob(oof, cal).mean(axis=0)
    thr = best_threshold(y_tr, oof_p)
    cv_metrics = evaluate(y_tr, oof_p, thr)

    n_rounds = int(np.mean(iters))
    booster = fit(params, X_tr, y_tr, rounds=n_rounds)
    p_te = to_prob(booster.predict(X_te, raw_score=True), cal)
    test_metrics = evaluate(y_te, p_te, thr)

    contrib = booster.predict(X_tr, pred_contrib=True)[:, :-1] * cal["a"]
    importance = pd.Series(np.abs(contrib).mean(axis=0), index=feats).sort_values(ascending=False)

    base_rate = float(y_tr.mean())
    return {
        "years_ahead": h,
        "booster": booster.model_to_string(),
        "meta": {
            "years_ahead": h,
            "source_file": file_for_horizon(h),
            "model_file": f"model_{h}y.txt",
            "features": feats,
            "threshold": thr,
            "calibration": cal,
            "risk_bands": {"medium": round(thr / 2, 3), "high": thr},
            "base_rate": base_rate,
            "params": {**params, "num_boost_round": n_rounds},
            "importance": importance.round(5).to_dict(),
            "feature_stats": {f: {"median": float(X_tr[f].median()),
                                  "p05": float(X_tr[f].quantile(0.05)),
                                  "p95": float(X_tr[f].quantile(0.95)),
                                  "missing_pct": float(X_tr[f].isna().mean() * 100)} for f in feats},
            "n_train": int(len(y_tr)), "n_train_bankrupt": int(y_tr.sum()),
            "n_test": int(len(y_te)), "n_test_bankrupt": int(y_te.sum()),
            "metrics": {"cv": cv_metrics, "test": test_metrics,
                        "test_ci95": bootstrap_ci(y_te, p_te, thr),
                        "cv_pr_auc_during_tuning": cv_ap_tuning},
            "examples": pick_examples(X_te, y_te, p_te, thr),
        },
        "full_ranking": ranking.round(2).to_dict(),
        "test_y": y_te.tolist(), "test_p": p_te.tolist(),
        "oof_y": y_tr.tolist(), "oof_p": oof_p.tolist(),
    }


# ----------------------------------------------------------------------------- main
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=N_TRIALS, help="Optuna trials per horizon")
    ap.add_argument("--jobs", type=int, default=len(HORIZONS), help="horizons trained in parallel")
    args = ap.parse_args()

    results = Parallel(n_jobs=args.jobs)(delayed(train_horizon)(h, args.trials) for h in HORIZONS)
    results = sorted(results, key=lambda r: r["years_ahead"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        # LightGBM cannot read its model file back with Windows (CRLF) line endings
        (MODELS_DIR / r["meta"]["model_file"]).write_text(r["booster"], encoding="utf-8", newline="\n")

    metadata = {
        "model_version": "2.0",
        "trained_on": date.today().isoformat(),
        "dataset": "Polish Companies Bankruptcy (UCI ML Repository, Tomczak 2016)",
        "algorithm": "LightGBM gradient-boosted decision trees",
        "libraries": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                      "scikit-learn": sklearn.__version__, "optuna": optuna.__version__},
        "horizons": {str(r["years_ahead"]): r["meta"] for r in results},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default),
                                              encoding="utf-8")
    (REPORTS_DIR / "predictions.json").write_text(json.dumps(
        {r["years_ahead"]: {k: r[k] for k in ("test_y", "test_p", "oof_y", "oof_p", "full_ranking")}
         for r in results}, default=_json_default), encoding="utf-8")

    from .report import build_reports
    build_reports(metadata)

    print(f"\n{'Horizon':<10}{'Threshold':>10}{'Test F1':>9}{'Test PR-AUC':>13}{'Test ROC-AUC':>14}{'CV PR-AUC':>11}")
    for r in results:
        m = r["meta"]
        t = m["metrics"]["test"]
        print(f"{str(m['years_ahead']) + ' year(s)':<10}{m['threshold']:>10.2f}{t['f1']:>9.3f}"
              f"{t['pr_auc']:>13.3f}{t['roc_auc']:>14.3f}{m['metrics']['cv']['pr_auc']:>11.3f}")


if __name__ == "__main__":
    main()
