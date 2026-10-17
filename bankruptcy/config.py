"""Project-wide settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

HORIZONS = [1, 2, 3, 4, 5]  # years ahead

RANDOM_STATE = 42
TEST_SIZE = 0.15
CV_FOLDS = 5
N_FEATURES = 20        # ratios kept per model: as accurate as all 64 in CV, keeps the app form short
N_TRIALS = 40          # Optuna hyper-parameter trials per horizon


def file_for_horizon(years_ahead: int) -> str:
    """UCI: `{k}year.arff` holds ratios from year k of the forecasting period and the
    bankruptcy status after (6 - k) years. So 1 year ahead -> 5year.arff, 5 years ahead -> 1year.arff."""
    return f"{6 - years_ahead}year.arff"
