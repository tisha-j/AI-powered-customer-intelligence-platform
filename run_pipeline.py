from src.data_pipeline import load_raw_data, clean_transactions
from src.feature_engineering import build_customer_features
from src.segmentation import run_segmentation
from src.churn_model import train_models
from src.explainability import create_shap_report

def main():
    print("\n=== 1. LOAD DATA ===")
    raw = load_raw_data()

    print("\n=== 2. CLEAN DATA ===")
    transactions = clean_transactions(raw)

    print("\n=== 3. CUSTOMER FEATURES ===")
    features, _ = build_customer_features(transactions)

    print("\n=== 4. CUSTOMER SEGMENTATION ===")
    segmented = run_segmentation(features)

    print("\n=== 5. RISK MODEL ===")
    model_results = train_models(segmented)

    print("\n=== 6. SHAP EXPLANATION ===")
    create_shap_report(
        segmented,
        model_results["model"],
        model_results["feature_columns"]
    )

    print("\nPipeline completed.")

if __name__ == "__main__":
    main()
