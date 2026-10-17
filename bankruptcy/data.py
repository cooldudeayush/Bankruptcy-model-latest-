"""Loading the Polish Companies Bankruptcy data (UCI) and splitting it."""
from pathlib import Path

import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, RANDOM_STATE, TEST_SIZE, file_for_horizon
from .features import RATIO_COLUMNS


def load_arff(path: Path) -> pd.DataFrame:
    """Read one .arff file into a DataFrame with float ratios and an int `class` column."""
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    df["class"] = df["class"].map(lambda v: int(v.decode() if isinstance(v, bytes) else v))
    df[RATIO_COLUMNS] = df[RATIO_COLUMNS].astype(float)
    return df


def load_horizon(years_ahead: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return load_arff(data_dir / file_for_horizon(years_ahead))


def train_test(df: pd.DataFrame):
    """Stratified 85/15 split. Uses the same seed as the original notebook,
    so the test rows are identical and old and new models can be compared fairly."""
    X, y = df[RATIO_COLUMNS], df["class"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return (X_tr.reset_index(drop=True), X_te.reset_index(drop=True),
            y_tr.reset_index(drop=True), y_te.reset_index(drop=True))
