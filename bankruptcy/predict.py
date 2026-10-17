"""Load the trained models and turn financial ratios into a bankruptcy risk score.

    predictor = BankruptcyPredictor()
    result = predictor.predict(1, {"Attr27": 1.2, "Attr46": 0.8, ...})
    reasons = predictor.explain(1, {...})
"""
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import MODELS_DIR
from .features import meaning


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class HorizonModel:
    years_ahead: int
    features: list
    threshold: float
    calibration: dict      # {"a": slope, "b": intercept} applied to the raw log-odds
    risk_bands: dict       # {"medium": p, "high": p}
    medians: dict          # typical value of each feature in the training data
    metrics: dict
    booster: lgb.Booster

    def _frame(self, data, fill_missing=False) -> pd.DataFrame:
        """Missing values: by default they are passed to LightGBM as missing. In this dataset a value
        missing from the accounts is itself a warning sign (e.g. no prior-year sales). When the value is
        simply unknown to the user, `fill_missing=True` uses the typical (median) value instead."""
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        for f in self.features:
            if f not in df.columns:
                df[f] = np.nan
        X = df[self.features].apply(pd.to_numeric, errors="coerce").astype(float)
        return X.fillna(self.medians) if fill_missing else X

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X, raw_score=True)

    def probability(self, data, fill_missing=False) -> np.ndarray:
        X = self._frame(data, fill_missing)
        z = self.raw_score(X)
        return _sigmoid(self.calibration["a"] * z + self.calibration["b"])

    def risk_level(self, p: float) -> str:
        if p >= self.risk_bands["high"]:
            return "High"
        if p >= self.risk_bands["medium"]:
            return "Medium"
        return "Low"


class BankruptcyPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        models_dir = Path(models_dir)
        self.metadata = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
        self.models = {}
        for key, m in self.metadata["horizons"].items():
            self.models[int(key)] = HorizonModel(
                years_ahead=int(key),
                features=m["features"],
                threshold=m["threshold"],
                calibration=m["calibration"],
                risk_bands=m["risk_bands"],
                medians={f: s["median"] for f, s in m["feature_stats"].items()},
                metrics=m["metrics"],
                booster=lgb.Booster(model_file=str(models_dir / m["model_file"])),
            )

    @property
    def horizons(self):
        return sorted(self.models)

    def features(self, years_ahead: int) -> list:
        return self.models[years_ahead].features

    def predict(self, years_ahead: int, data, fill_missing=False) -> pd.DataFrame:
        """Score one company (dict) or many (DataFrame). Returns probability, verdict and risk level."""
        m = self.models[years_ahead]
        p = m.probability(data, fill_missing)
        return pd.DataFrame({
            "bankruptcy_probability": p,
            "threshold": m.threshold,
            "prediction": np.where(p >= m.threshold, "Bankrupt", "Not bankrupt"),
            "risk_level": [m.risk_level(v) for v in p],
        })

    def explain(self, years_ahead: int, data: dict, fill_missing=False) -> pd.DataFrame:
        """Per-feature contributions (SHAP values computed by LightGBM) for one company.

        `contribution` is in log-odds: positive pushes towards bankruptcy, negative away from it.
        The contributions plus `base` add up to the company's score.
        """
        m = self.models[years_ahead]
        X = m._frame(data, fill_missing)
        contrib = m.booster.predict(X, pred_contrib=True)[0] * m.calibration["a"]
        out = pd.DataFrame({
            "feature": m.features,
            "meaning": [meaning(f) for f in m.features],
            "value": X.iloc[0].values,
            "contribution": contrib[:-1],
        })
        out.attrs["base"] = float(contrib[-1] + m.calibration["b"])
        return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
