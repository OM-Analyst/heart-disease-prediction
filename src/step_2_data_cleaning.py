import pandas as pd
from pathlib import Path


def main():
    # File paths
    input_path = Path("data/heart_disease.csv")
    output_data_path = Path("data/heart_disease_cleaned.csv")
    output_dir = Path("outputs/step_2_data_cleaning")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_path)

    print("\n==============================")
    print("STEP 2: DATA CLEANING")
    print("==============================")

    # Original dataset shape
    original_rows, original_columns = df.shape
    print(f"\nOriginal dataset shape: {original_rows} rows and {original_columns} columns")

    # Check missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()

    print("\nMissing values per column:")
    print(missing_values)
    print(f"\nTotal missing values: {total_missing}")

    # Check duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate rows found: {duplicate_count}")

    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    cleaned_rows, cleaned_columns = df_cleaned.shape
    rows_removed = original_rows - cleaned_rows

    print(f"\nCleaned dataset shape: {cleaned_rows} rows and {cleaned_columns} columns")
    print(f"Rows removed: {rows_removed}")

    # Check target distribution after cleaning
    target_counts = df_cleaned["target"].value_counts()
    target_percentages = df_cleaned["target"].value_counts(normalize=True) * 100

    print("\nTarget distribution after cleaning:")
    print(target_counts)

    print("\nTarget distribution percentage after cleaning:")
    print(target_percentages.round(2))

    # Save cleaned dataset
    df_cleaned.to_csv(output_data_path, index=False)

    # Save reports
    missing_values.to_csv(output_dir / "missing_values_after_cleaning_check.csv")
    target_counts.to_csv(output_dir / "target_distribution_after_cleaning.csv")
    target_percentages.round(2).to_csv(output_dir / "target_distribution_percentage_after_cleaning.csv")

    with open(output_dir / "data_cleaning_report.txt", "w") as file:
        file.write("STEP 2: DATA CLEANING REPORT\n")
        file.write("============================\n\n")
        file.write(f"Original dataset shape: {original_rows} rows and {original_columns} columns\n")
        file.write(f"Duplicate rows found: {duplicate_count}\n")
        file.write(f"Rows removed: {rows_removed}\n")
        file.write(f"Cleaned dataset shape: {cleaned_rows} rows and {cleaned_columns} columns\n\n")
        file.write("Missing values per column:\n")
        file.write(str(missing_values))
        file.write(f"\n\nTotal missing values: {total_missing}\n\n")
        file.write("Target distribution after cleaning:\n")
        file.write(str(target_counts))
        file.write("\n\nTarget distribution percentage after cleaning:\n")
        file.write(str(target_percentages.round(2)))

    print("\nStep 2 completed successfully.")
    print(f"Cleaned dataset saved as: {output_data_path}")
    print(f"Reports saved in: {output_dir}")


if __name__ == "__main__":
    main()