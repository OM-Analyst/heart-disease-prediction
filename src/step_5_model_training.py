import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib


def evaluate_model(model_name, model, X_test, y_test):
    """
    Evaluate a trained model and return key classification metrics.
    """

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc
    }


def main():
    # File paths
    processed_dir = Path("data/processed")
    output_dir = Path("outputs/step_5_model_training")
    model_dir = Path("models")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load processed datasets
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    print("\n==============================")
    print("STEP 5: MODEL TRAINING")
    print("==============================")

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []

    # Train and evaluate models
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        # Save each trained model
        model_filename = model_name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, model_dir / model_filename)

        # Evaluate model
        metrics = evaluate_model(model_name, model, X_test, y_test)
        results.append(metrics)

        print(f"{model_name} completed.")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    # Convert results to dataframe
    results_df = pd.DataFrame(results)

    # Sort by recall first because recall is important in healthcare
    results_df = results_df.sort_values(by="recall", ascending=False)

    # Save model comparison
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)

    # Select best model based on recall
    best_model_name = results_df.iloc[0]["model"]
    best_model_filename = best_model_name.lower().replace(" ", "_") + ".pkl"

    best_model = joblib.load(model_dir / best_model_filename)
    joblib.dump(best_model, model_dir / "best_model.pkl")

    # Save report
    with open(output_dir / "model_training_report.txt", "w") as file:
        file.write("STEP 5: MODEL TRAINING REPORT\n")
        file.write("=============================\n\n")

        file.write("Models trained:\n")
        for model_name in models.keys():
            file.write(f"- {model_name}\n")

        file.write("\nModel comparison:\n")
        file.write(results_df.to_string(index=False))

        file.write("\n\nBest model selected based on recall:\n")
        file.write(str(best_model_name))

        file.write("\n\nWhy recall was prioritized:\n")
        file.write(
            "Recall is important in healthcare because the model should reduce the chance "
            "of missing patients who actually have heart disease."
        )

    print("\nStep 5 completed successfully.")
    print(f"Model comparison saved in: {output_dir}")
    print(f"Trained models saved in: {model_dir}")
    print(f"Best model selected: {best_model_name}")


if __name__ == "__main__":
    main()