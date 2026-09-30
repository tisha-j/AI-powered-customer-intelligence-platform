import pandas as pd
from src.feature_engineering import build_customer_features

def test_customer_features(tmp_path):
    data = pd.DataFrame({
        "Customer ID": [1, 1, 2, 2],
        "InvoiceDate": pd.to_datetime([
            "2011-01-01",
            "2011-01-10",
            "2011-01-05",
            "2011-01-20"
        ]),
        "Invoice": ["A", "B", "C", "D"],
        "StockCode": ["X", "Y", "X", "Z"],
        "Revenue": [10, 20, 30, 40],
        "Quantity": [1, 2, 3, 4],
        "Country": ["UK", "UK", "UK", "UK"]
    })

    features, _ = build_customer_features(data)

    assert len(features) == 2
    assert "Recency" in features.columns
    assert "Monetary" in features.columns
    assert "Risk" in features.columns
