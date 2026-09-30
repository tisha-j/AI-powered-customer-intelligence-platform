import shap
import matplotlib.pyplot as plt

from .config import REPORT_DIR


def create_shap_report(df, model, feature_columns):

    X = df[feature_columns].fillna(0)

    sample = X.sample(
        min(500, len(X)),
        random_state=42
    )

    try:

        # Handle sklearn pipeline
        if hasattr(model, "named_steps"):

            explainer = shap.Explainer(
                model.predict_proba,
                sample
            )

            shap_values = explainer(sample)

        else:

            explainer = shap.TreeExplainer(model)

            shap_values = explainer(
                sample
            )

        # Binary classification:
        # use positive-class SHAP values when necessary.
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        plt.figure()

        shap.plots.bar(
            shap_values,
            max_display=15,
            show=False
        )

        plt.tight_layout()

        output = (
            REPORT_DIR
            / "shap_feature_importance.png"
        )

        plt.savefig(
            output,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"SHAP report saved: {output}"
        )

    except Exception as exc:

        print(
            f"SHAP report failed: {exc}"
        )