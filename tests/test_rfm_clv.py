"""
Tests for RFM, customer segmentation, and CLV outputs from Notebook 04.

Notably, test_monetary_is_not_degenerate and test_segments_are_not_collapsed
are written specifically to catch the two bugs found during manual review of
this project: a leftover column-mapping patch that silently zeroed out every
customer's Monetary value, which in turn collapsed every cluster into a single
"At Risk" label. Both bugs previously passed unnoticed because the notebook
ran without raising any exception - these tests exist so that class of failure
is caught automatically instead of requiring a manual read-through of the output.
"""


def test_monetary_is_not_degenerate(rfm_data):
    assert "Monetary" in rfm_data.columns
    assert rfm_data["Monetary"].std() > 0, (
        "Monetary has zero variance - this usually means the sales column mapping "
        "used to compute Monetary_Value silently produced all zeros"
    )
    assert rfm_data["Monetary"].sum() > 0, "Total Monetary value should not be zero"


def test_recency_and_frequency_are_positive(rfm_data):
    assert (rfm_data["Recency"] >= 0).all(), "Recency (days) cannot be negative"
    assert (rfm_data["Frequency"] > 0).all(), "Every customer should have at least 1 order"


def test_segments_are_not_collapsed(customer_segments):
    segment_col = "Cluster_Name" if "Cluster_Name" in customer_segments.columns else "Customer_Segment"
    unique_segments = customer_segments[segment_col].nunique()
    assert unique_segments > 1, (
        f"All customers were assigned to a single '{segment_col}' value - "
        "this usually means the segment-naming logic collapsed due to a "
        "degenerate feature (e.g. Monetary = 0 for every cluster)"
    )


def test_clv_is_positive(customer_clv):
    assert "CLV_Simple" in customer_clv.columns
    assert (customer_clv["CLV_Simple"] >= 0).all(), "CLV should never be negative"
    assert customer_clv["CLV_Simple"].std() > 0, "CLV has zero variance across customers"


def test_customer_ids_consistent_across_rfm_and_clv(rfm_data, customer_clv):
    rfm_ids = set(rfm_data["Customer_ID"])
    clv_ids = set(customer_clv["Customer_ID"])
    overlap = rfm_ids & clv_ids
    assert len(overlap) > 0, "RFM and CLV customer IDs do not overlap at all - check the join keys"
