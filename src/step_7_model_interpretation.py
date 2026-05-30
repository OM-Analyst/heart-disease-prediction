import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

from sklearn.inspection import permutation_importance


def main():
    # File paths
    processed_dir = Path("data/processed")
    model_path = Path("models/best_model.pkl")
    output_dir = Path("outputs/step_7_model_interpretation")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    # Load best model
    model = joblib.load(model_path)

    print("\n================================")
    print("STEP 7: MODEL INTERPRETATION")
    print("================================")

    print(f"\nModel used: {model_path}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # ---------------------------------------------------------
    # 1. Permutation Importance
    # ---------------------------------------------------------
    # This checks how much model performance drops when each feature is shuffled.
    # If performance drops a lot, that feature is important.
    permutation_result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="recall"
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": permutation_result.importances_mean,
        "importance_std": permutation_result.importances_std
    })

    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False
    )

    # Save feature importance table
    importance_df.to_csv(
        output_dir / "permutation_feature_importance.csv",
        index=False
    )

    # Plot top 10 important features
    top_features = importance_df.head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_features,
        x="importance_mean",
        y="feature"
    )
    plt.title("Top 10 Important Features for Heart Disease Prediction")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "top_10_feature_importance.png")
    plt.close()

    # ---------------------------------------------------------
    # 2. Built-in Feature Importance if model supports it
    # ---------------------------------------------------------
    if hasattr(model, "feature_importances_"):
        built_in_importance_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance": model.feature_importances_
        })

        built_in_importance_df = built_in_importance_df.sort_values(
            by="importance",
            ascending=False
        )

        built_in_importance_df.to_csv(
            output_dir / "built_in_feature_importance.csv",
            index=False
        )

        top_built_in = built_in_importance_df.head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=top_built_in,
            x="importance",
            y="feature"
        )
        plt.title("Top 10 Built-in Feature Importances")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(output_dir / "top_10_built_in_feature_importance.png")
        plt.close()

    # ---------------------------------------------------------
    # 3. Save interpretation report
    # ---------------------------------------------------------
    top_5_features = importance_df.head(5)

    with open(output_dir / "model_interpretation_report.txt", "w") as file:
        file.write("STEP 7: MODEL INTERPRETATION REPORT\n")
        file.write("===================================\n\n")

        file.write(f"Model used: {model_path}\n")
        file.write(f"X_test shape: {X_test.shape}\n")
        file.write(f"y_test shape: {y_test.shape}\n\n")

        file.write("Interpretation Method Used:\n")
        file.write(
            "Permutation importance was used to identify which features had the "
            "strongest influence on the model's heart disease predictions.\n\n"
        )

        file.write("Top 5 Important Features:\n")
        for index, row in top_5_features.iterrows():
            file.write(
                f"- {row['feature']}: {row['importance_mean']:.4f}\n"
            )

        file.write("\nHealth Informatics Interpretation:\n")
        file.write(
            "The most important features represent clinical indicators that contributed "
            "most strongly to the model's prediction of heart disease. These may include "
            "factors such as chest pain type, maximum heart rate, exercise-induced angina, "
            "ST depression, number of major vessels, or thalassemia results depending on "
            "the final trained model.\n\n"
        )

        file.write(
            "This step helps make the prediction system more explainable. In a health "
            "informatics setting, explainability is important because healthcare workers "
            "need to understand which patient factors are influencing risk predictions."
        )

    print("\nTop 10 important features:")
    print(top_features)

    print("\nStep 7 completed successfully.")
    print(f"Interpretation outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()