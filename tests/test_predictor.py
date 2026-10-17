import math

import numpy as np
import pandas as pd
import pytest

from bankruptcy.config import HORIZONS, file_for_horizon
from bankruptcy.features import RATIO_COLUMNS
from bankruptcy.predict import BankruptcyPredictor


@pytest.fixture(scope="module")
def predictor():
    return BankruptcyPredictor()


def test_horizon_to_file_mapping():
    # UCI: 5year.arff = bankruptcy status after 1 year ... 1year.arff = after 5 years
    assert [file_for_horizon(h) for h in HORIZONS] == [
        "5year.arff", "4year.arff", "3year.arff", "2year.arff", "1year.arff"]


def test_every_horizon_has_a_model(predictor):
    assert predictor.horizons == HORIZONS
    for h in HORIZONS:
        m = predictor.models[h]
        assert 0 < m.threshold < 1
        assert set(m.features) <= set(RATIO_COLUMNS)
        assert m.booster.num_feature() == len(m.features)


def test_examples_are_scored_sensibly(predictor):
    for h in HORIZONS:
        ex = predictor.metadata["horizons"][str(h)]["examples"]
        safe = predictor.predict(h, ex["Healthy company"]["values"]).iloc[0]
        risky = predictor.predict(h, ex["Distressed company"]["values"]).iloc[0]
        assert safe["prediction"] == "Not bankrupt" and safe["risk_level"] == "Low"
        assert risky["prediction"] == "Bankrupt" and risky["risk_level"] == "High"
        assert risky["bankruptcy_probability"] > safe["bankruptcy_probability"]


def test_missing_values_are_allowed(predictor):
    out = predictor.predict(1, {})
    p = out["bankruptcy_probability"].iloc[0]
    assert 0 <= p <= 1 and not math.isnan(p)
    partial = predictor.predict(1, {predictor.features(1)[0]: 0.5})
    assert 0 <= partial["bankruptcy_probability"].iloc[0] <= 1


def test_unknown_values_filled_with_typical_values_look_safe(predictor):
    # an empty form means "I don't know", which must not look like a distressed company
    for h in HORIZONS:
        r = predictor.predict(h, {}, fill_missing=True).iloc[0]
        assert r["risk_level"] == "Low", h


def test_batch_prediction_keeps_row_order(predictor):
    ex = predictor.metadata["horizons"]["2"]["examples"]
    df = pd.DataFrame([ex["Healthy company"]["values"], ex["Distressed company"]["values"]])
    df["unrelated_column"] = "ignored"
    out = predictor.predict(2, df)
    assert list(out["prediction"]) == ["Not bankrupt", "Bankrupt"]


def test_explanation_adds_up_to_the_probability(predictor):
    values = predictor.metadata["horizons"]["3"]["examples"]["Borderline company"]["values"]
    ex = predictor.explain(3, values)
    z = ex["contribution"].sum() + ex.attrs["base"]
    p = predictor.predict(3, values)["bankruptcy_probability"].iloc[0]
    assert np.isclose(1 / (1 + np.exp(-z)), p, atol=1e-6)


def test_new_models_beat_v1_on_average(predictor):
    from bankruptcy.report import LEGACY_V1
    for key in ("pr_auc", "f1"):
        new = np.mean([predictor.models[h].metrics["test"][key] for h in HORIZONS])
        old = np.mean([LEGACY_V1[h][key] for h in HORIZONS])
        assert new > old, key
