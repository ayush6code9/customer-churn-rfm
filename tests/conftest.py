"""
Shared pytest fixtures for the Retail & Marketing Analytics test suite.

Run from the project root with:
    pytest tests/ -v

Tests are read-only: they check the outputs already produced by Notebooks 01-06
rather than re-running the pipeline, so they run in seconds and can be wired into
CI (e.g. GitHub Actions) to catch pipeline regressions before merge.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_MODELS = PROJECT_ROOT / "outputs" / "models"
OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"


def _load_csv_or_skip(path: Path) -> pd.DataFrame:
    """Load a CSV, or skip the test with a clear message if the notebook that
    produces it hasn't been run yet."""
    if not path.exists():
        pytest.skip(f"{path} not found - run the notebook that generates it first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def cleaned_sales():
    return _load_csv_or_skip(DATA_PROCESSED / "cleaned_retail_sales.csv")


@pytest.fixture(scope="session")
def rfm_data():
    return _load_csv_or_skip(DATA_PROCESSED / "rfm_analysis.csv")


@pytest.fixture(scope="session")
def customer_segments():
    return _load_csv_or_skip(DATA_PROCESSED / "customer_segments.csv")


@pytest.fixture(scope="session")
def customer_clv():
    return _load_csv_or_skip(DATA_PROCESSED / "customer_clv.csv")


@pytest.fixture(scope="session")
def churn_features():
    return _load_csv_or_skip(DATA_PROCESSED / "churn_features.csv")


@pytest.fixture(scope="session")
def churn_predictions():
    return _load_csv_or_skip(DATA_PROCESSED / "churn_predictions.csv")


@pytest.fixture(scope="session")
def churn_model_info():
    path = OUTPUTS_MODELS / "churn_model_info.txt"
    if not path.exists():
        pytest.skip(f"{path} not found - run Notebook 06 first")
    return path.read_text()
