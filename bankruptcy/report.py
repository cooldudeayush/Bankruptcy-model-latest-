"""Write reports/metrics.json, reports/metrics.md and the charts in reports/figures/."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from .config import FIGURES_DIR, REPORTS_DIR
from .features import meaning

# v1 models (Final_Master_Code.ipynb) scored on the same test rows, by years ahead
LEGACY_V1 = {
    1: {"f1": 0.739, "precision": 0.837, "recall": 0.661, "roc_auc": 0.959, "pr_auc": 0.812, "threshold": 0.41},
    2: {"f1": 0.694, "precision": 0.746, "recall": 0.649, "roc_auc": 0.914, "pr_auc": 0.700, "threshold": 0.16},
    3: {"f1": 0.519, "precision": 0.596, "recall": 0.459, "roc_auc": 0.916, "pr_auc": 0.567, "threshold": 0.19},
    4: {"f1": 0.612, "precision": 0.789, "recall": 0.500, "roc_auc": 0.913, "pr_auc": 0.640, "threshold": 0.30},
    5: {"f1": 0.704, "precision": 0.833, "recall": 0.610, "roc_auc": 0.946, "pr_auc": 0.749, "threshold": 0.60},
}

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
OLD = "#c3c2b7"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "legend.frameon": False,
        "legend.fontsize": 9, "text.color": INK,
    })


def plot_comparison(meta):
    hs = sorted(int(h) for h in meta["horizons"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in [(axes[0], "f1", "F1 score at the chosen threshold"),
                           (axes[1], "pr_auc", "PR-AUC (average precision)")]:
        old = [LEGACY_V1[h][key] for h in hs]
        new = [meta["horizons"][str(h)]["metrics"]["test"][key] for h in hs]
        x = np.arange(len(hs))
        ax.bar(x - 0.17, old, 0.32, color=OLD, label="Old models (v1)")
        ax.bar(x + 0.17, new, 0.32, color=SERIES[0], label="New models (v2)")
        for xi, v in zip(x, new):
            ax.text(xi + 0.17, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x, [f"{h} yr" for h in hs])
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score on the held-out test set")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES_DIR / "old_vs_new.png", dpi=160)
    plt.close(fig)


def plot_importance(meta):
    for h, m in meta["horizons"].items():
        imp = list(m["importance"].items())[:10][::-1]
        labels = [f"{meaning(f)} ({f})" for f, _ in imp]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.barh(range(len(imp)), [v for _, v in imp], 0.6, color=SERIES[0])
        ax.set_yticks(range(len(imp)), labels, fontsize=8.5, color=INK2)
        ax.set_xlabel("Average impact on the risk score (mean |SHAP|, log-odds)")
        ax.set_title(f"Top 10 ratios driving the {h}-year-ahead prediction", loc="left")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"importance_{h}y.png", dpi=160)
        plt.close(fig)


def plot_pr_curves(meta, preds):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for i, h in enumerate(sorted(preds, key=int)):
        y, p = np.array(preds[h]["test_y"]), np.array(preds[h]["test_p"])
        prec, rec, _ = precision_recall_curve(y, p)
        ap = meta["horizons"][str(h)]["metrics"]["test"]["pr_auc"]
        ax.plot(rec, prec, color=SERIES[i], lw=2, label=f"{h} yr ahead (PR-AUC {ap:.2f})")
    ax.set_xlabel("Recall: share of bankrupt companies caught")
    ax.set_ylabel("Precision: share of alarms that were right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curves on the test set", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curves.png", dpi=160)
    plt.close(fig)


def plot_calibration(preds):
    y = np.concatenate([preds[h]["oof_y"] for h in preds])
    p = np.concatenate([preds[h]["oof_p"] for h in preds])
    bins = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0])
    idx = np.digitize(p, bins[1:-1])
    mp = [p[idx == i].mean() for i in range(len(bins) - 1) if (idx == i).sum() > 20]
    my = [y[idx == i].mean() for i in range(len(bins) - 1) if (idx == i).sum() > 20]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, label="Perfect calibration")
    ax.plot(mp, my, color=SERIES[0], lw=2, marker="o", ms=6, mec=SURFACE, mew=2, label="Model")
    ax.set_xlabel("Predicted probability of bankruptcy")
    ax.set_ylabel("Share that actually went bankrupt")
    ax.set_title("Are the probabilities honest?", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=160)
    plt.close(fig)


def write_markdown(meta):
    lines = ["# Model performance (v2)", "",
             f"Trained {meta['trained_on']} · {meta['algorithm']} · LightGBM {meta['libraries']['lightgbm']}", "",
             "Scores on the held-out 15% test set (never used for training, feature selection, tuning or thresholds). "
             "95% bootstrap intervals in brackets.", "",
             "| Years ahead | Data file | Features | Threshold | F1 | Precision | Recall | ROC-AUC | PR-AUC |",
             "|---|---|---|---|---|---|---|---|---|"]
    for h in sorted(meta["horizons"], key=int):
        m = meta["horizons"][h]
        t, ci = m["metrics"]["test"], m["metrics"]["test_ci95"]
        lines.append(f"| {h} | {m['source_file']} | {len(m['features'])} | {m['threshold']:.2f} | "
                     f"{t['f1']:.3f} [{ci['f1'][0]:.2f}–{ci['f1'][1]:.2f}] | {t['precision']:.3f} | {t['recall']:.3f} | "
                     f"{t['roc_auc']:.3f} [{ci['roc_auc'][0]:.2f}–{ci['roc_auc'][1]:.2f}] | "
                     f"{t['pr_auc']:.3f} [{ci['pr_auc'][0]:.2f}–{ci['pr_auc'][1]:.2f}] |")
    lines += ["", "## Compared with the v1 models (same test rows)", "",
              "| Years ahead | v1 F1 | v2 F1 | v1 PR-AUC | v2 PR-AUC | v1 ROC-AUC | v2 ROC-AUC |",
              "|---|---|---|---|---|---|---|"]
    for h in sorted(meta["horizons"], key=int):
        t, o = meta["horizons"][h]["metrics"]["test"], LEGACY_V1[int(h)]
        lines.append(f"| {h} | {o['f1']:.3f} | {t['f1']:.3f} | {o['pr_auc']:.3f} | {t['pr_auc']:.3f} | "
                     f"{o['roc_auc']:.3f} | {t['roc_auc']:.3f} |")
    avg = lambda k, src: np.mean([src(h)[k] for h in meta["horizons"]])
    lines += [f"| **Average** | {avg('f1', lambda h: LEGACY_V1[int(h)]):.3f} | "
              f"{avg('f1', lambda h: meta['horizons'][h]['metrics']['test']):.3f} | "
              f"{avg('pr_auc', lambda h: LEGACY_V1[int(h)]):.3f} | "
              f"{avg('pr_auc', lambda h: meta['horizons'][h]['metrics']['test']):.3f} | "
              f"{avg('roc_auc', lambda h: LEGACY_V1[int(h)]):.3f} | "
              f"{avg('roc_auc', lambda h: meta['horizons'][h]['metrics']['test']):.3f} |", "",
              "Note: v1 chose its features using the whole dataset, test rows included, so its test scores are "
              "slightly optimistic. v2 uses training rows only.", "",
              "## Cross-validated scores (5-fold × 3 repeats on the training part)", "",
              "| Years ahead | F1 | PR-AUC | ROC-AUC |", "|---|---|---|---|"]
    for h in sorted(meta["horizons"], key=int):
        c = meta["horizons"][h]["metrics"]["cv"]
        lines.append(f"| {h} | {c['f1']:.3f} | {c['pr_auc']:.3f} | {c['roc_auc']:.3f} |")
    lines += ["", "![Old vs new](figures/old_vs_new.png)", "", "![PR curves](figures/pr_curves.png)", "",
              "![Calibration](figures/calibration.png)", ""]
    (REPORTS_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def build_reports(meta):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preds = json.loads((REPORTS_DIR / "predictions.json").read_text(encoding="utf-8"))
    _style()
    plot_comparison(meta)
    plot_importance(meta)
    plot_pr_curves(meta, preds)
    plot_calibration(preds)
    summary = {h: {"test": m["metrics"]["test"], "test_ci95": m["metrics"]["test_ci95"],
                   "cv": m["metrics"]["cv"], "legacy_v1_test": LEGACY_V1[int(h)]}
               for h, m in meta["horizons"].items()}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(meta)


if __name__ == "__main__":
    from .config import MODELS_DIR
    build_reports(json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8")))
