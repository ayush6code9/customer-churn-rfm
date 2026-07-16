"""
Tests for the churn prediction model and its artifacts from Notebook 06.
"""

import re
from pathlib import Path

import joblib
import pandas as pd

from conftest import OUTPUTS_MODELS


def test_no_recency_derived_leakage_columns(churn_features):
    """Recency (and anything computed from it, since it defines the churn label)
    must never appear as a model feature - otherwise the model would be reading
    the answer directly off a feature instead of learning from behavior."""
    leaking_columns = {"Recency", "R_Score", "RFM_Score", "RFM_Score_Numeric",
                        "Customer_Segment", "Cluster", "Cluster_Name"}
    present_leaks = leaking_columns & set(churn_features.columns)
    assert not present_leaks, f"Leakage columns found in churn feature table: {present_leaks}"


def test_churned_target_is_binary(churn_features):
    assert "Churned" in churn_features.columns
    assert set(churn_features["Churned"].unique()).issubset({0, 1})


def test_model_artifacts_exist_and_load():
    model_path = OUTPUTS_MODELS / "churn_model_best.pkl"
    if not model_path.exists():
        model_path = OUTPUTS_MODELS / "churn_model_rf.pkl"
    scaler_path = OUTPUTS_MODELS / "churn_scaler.pkl"
    cols_path = OUTPUTS_MODELS / "churn_feature_columns.pkl"

    if not model_path.exists():
        import pytest
        pytest.skip("Churn model artifact not found - run Notebook 06 first")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(cols_path)

    assert hasattr(model, "predict_proba"), "Saved model must support predict_proba"
    assert hasattr(scaler, "transform"), "Saved scaler must support transform"
    assert isinstance(feature_cols, list) and len(feature_cols) > 0


def test_churn_probabilities_are_valid_range(churn_predictions):
    assert "Churn_Probability" in churn_predictions.columns
    assert churn_predictions["Churn_Probability"].between(0, 1).all(), (
        "Churn probabilities must fall between 0 and 1"
    )


def test_model_beats_minimum_auc_threshold(churn_model_info):
    """Guards against silently shipping a model that performs no better than
    chance. Threshold is intentionally modest (0.70) so it fails loudly only
    on a genuine regression, not on ordinary run-to-run variance."""
    match = re.search(r"Test ROC-AUC:\s*([\d.]+)", churn_model_info)
    assert match, "Could not find 'Test ROC-AUC' in churn_model_info.txt"
    auc = float(match.group(1))
    assert auc >= 0.70, f"Model test ROC-AUC ({auc:.3f}) is below the 0.70 minimum threshold"
