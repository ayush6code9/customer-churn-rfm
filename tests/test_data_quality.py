"""
Tests for the cleaned transaction dataset produced by Notebook 02.

These are the checks that would have caught a badly-mapped raw dataset before it
propagated errors downstream into RFM, clustering, and the churn model.
"""


def test_no_duplicate_rows(cleaned_sales):
    assert cleaned_sales.duplicated().sum() == 0, "Cleaned data should have no duplicate rows"


def test_no_nulls_in_critical_columns(cleaned_sales):
    critical_columns = [c for c in ["Order_ID", "Customer_ID", "Sales", "Quantity", "Order_Date"]
                         if c in cleaned_sales.columns]
    assert critical_columns, "None of the expected critical columns were found in the dataset"
    for col in critical_columns:
        assert cleaned_sales[col].isnull().sum() == 0, f"'{col}' should not contain nulls after cleaning"


def test_sales_are_non_negative(cleaned_sales):
    assert (cleaned_sales["Sales"] >= 0).all(), "Sales should never be negative"


def test_quantity_is_positive(cleaned_sales):
    assert (cleaned_sales["Quantity"] > 0).all(), "Quantity should always be greater than zero"


def test_discount_within_valid_range(cleaned_sales):
    if "Discount" in cleaned_sales.columns:
        assert cleaned_sales["Discount"].between(0, 1).all(), "Discount should be between 0 and 1"


def test_ship_date_not_before_order_date(cleaned_sales):
    if {"Order_Date", "Ship_Date"}.issubset(cleaned_sales.columns):
        import pandas as pd
        order_date = pd.to_datetime(cleaned_sales["Order_Date"])
        ship_date = pd.to_datetime(cleaned_sales["Ship_Date"])
        assert (ship_date >= order_date).all(), "Ship_Date should never be before Order_Date"
