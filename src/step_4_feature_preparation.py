import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def main():
    # File paths
    input_path = Path("data/heart_disease_cleaned.csv")
    processed_dir = Path("data/processed")
    output_dir = Path("outputs/step_4_feature_preparation")

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cleaned dataset
    df = pd.read_csv(input_path)

    print("\n====================================")
    print("STEP 4: FEATURE PREPARATION")
    print("====================================")

    print(f"\nDataset used: {input_path}")
    print(f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns")

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    print(f"\nFeature shape before split: {X.shape}")
    print(f"Target shape before split: {y.shape}")

    # Train-test split
    # stratify=y keeps the heart disease/no heart disease balance in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Numerical columns to scale
    numerical_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    # Scale numerical columns
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # Save processed datasets
    X_train_scaled.to_csv(processed_dir / "X_train.csv", index=False)
    X_test_scaled.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    # Save scaler for future use
    joblib.dump(scaler, processed_dir / "scaler.pkl")

    # Save feature names
    feature_names = pd.Series(X.columns)
    feature_names.to_csv(processed_dir / "feature_names.csv", index=False, header=["feature"])

    # Save report
    with open(output_dir / "feature_preparation_report.txt", "w") as file:
        file.write("STEP 4: FEATURE PREPARATION REPORT\n")
        file.write("==================================\n\n")
        file.write(f"Dataset used: {input_path}\n")
        file.write(f"Original cleaned dataset shape: {df.shape[0]} rows and {df.shape[1]} columns\n\n")

        file.write("Features used:\n")
        file.write(str(X.columns.tolist()))
        file.write("\n\n")

        file.write("Target variable:\n")
        file.write("target\n\n")

        file.write("Train-test split:\n")
        file.write("Training data: 80%\n")
        file.write("Testing data: 20%\n")
        file.write("Random state: 42\n")
        file.write("Stratified split: Yes\n\n")

        file.write(f"X_train shape: {X_train.shape}\n")
        file.write(f"X_test shape: {X_test.shape}\n")
        file.write(f"y_train shape: {y_train.shape}\n")
        file.write(f"y_test shape: {y_test.shape}\n\n")

        file.write("Numerical columns scaled:\n")
        file.write(str(numerical_columns))
        file.write("\n\n")

        file.write("Processed files saved in data/processed/.\n")

    print("\nStep 4 completed successfully.")
    print(f"Processed datasets saved in: {processed_dir}")
    print(f"Report saved in: {output_dir}")


if __name__ == "__main__":
    main()