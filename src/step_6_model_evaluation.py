import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


def main():
    # File paths
    processed_dir = Path("data/processed")
    model_path = Path("models/best_model.pkl")
    output_dir = Path("outputs/step_6_model_evaluation")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    # Load best model
    model = joblib.load(model_path)

    print("\n==============================")
    print("STEP 6: MODEL EVALUATION")
    print("==============================")

    print(f"\nModel used: {model_path}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Predict probabilities for ROC-AUC and ROC curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Heart Disease", "Heart Disease"],
        yticklabels=["No Heart Disease", "Heart Disease"]
    )
    plt.title("Confusion Matrix - Best Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    # ROC curve
    if y_prob is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"ROC Curve AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
        plt.title("ROC Curve - Best Model")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png")
        plt.close()

    # Classification report
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["No Heart Disease", "Heart Disease"],
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_dir / "classification_report.csv")

    # Save evaluation metrics
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        "Score": [accuracy, precision, recall, f1, roc_auc]
    })

    metrics_df.to_csv(output_dir / "evaluation_metrics.csv", index=False)

    # Save text report
    with open(output_dir / "model_evaluation_report.txt", "w") as file:
        file.write("STEP 6: MODEL EVALUATION REPORT\n")
        file.write("===============================\n\n")

        file.write(f"Model used: {model_path}\n")
        file.write(f"X_test shape: {X_test.shape}\n")
        file.write(f"y_test shape: {y_test.shape}\n\n")

        file.write("Evaluation Metrics:\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1-score: {f1:.4f}\n")

        if roc_auc is not None:
            file.write(f"ROC-AUC: {roc_auc:.4f}\n")

        file.write("\nConfusion Matrix:\n")
        file.write(str(cm))
        file.write("\n\nClassification Report:\n")
        file.write(classification_report(
            y_test,
            y_pred,
            target_names=["No Heart Disease", "Heart Disease"]
        ))

        file.write("\n\nHealth Informatics Interpretation:\n")
        file.write(
            "\nIn this heart disease prediction project, recall is especially important "
            "because it measures how well the model identifies patients who actually "
            "have heart disease. A high recall means the model is less likely to miss "
            "patients who may need further medical attention."
        )

    print("\nStep 6 completed successfully.")
    print(f"Evaluation outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()