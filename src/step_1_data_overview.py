import pandas as pd
from pathlib import Path


def main():
    # File paths
    data_path = Path("data/heart_disease.csv")
    output_dir = Path("outputs/step_1_data_overview")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)

    # Basic dataset information
    print("\n==============================")
    print("HEART DISEASE DATA OVERVIEW")
    print("==============================")

    print(f"\nDataset shape: {df.shape[0]} rows and {df.shape[1]} columns")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    print(missing_values)

    print("\nDuplicate rows:")
    duplicate_count = df.duplicated().sum()
    print(duplicate_count)

    print("\nTarget distribution:")
    target_counts = df["target"].value_counts()
    print(target_counts)

    print("\nTarget distribution percentage:")
    target_percentages = df["target"].value_counts(normalize=True) * 100
    print(target_percentages.round(2))

    print("\nSummary statistics:")
    summary_stats = df.describe()
    print(summary_stats)

    # Save outputs
    missing_values.to_csv(output_dir / "missing_values.csv")
    target_counts.to_csv(output_dir / "target_distribution.csv")
    target_percentages.round(2).to_csv(output_dir / "target_distribution_percentage.csv")
    summary_stats.to_csv(output_dir / "summary_statistics.csv")

    # Save a simple text report
    with open(output_dir / "data_overview_report.txt", "w") as file:
        file.write("HEART DISEASE DATA OVERVIEW\n")
        file.write("===========================\n\n")
        file.write(f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns\n\n")
        file.write("Column names:\n")
        file.write(str(df.columns.tolist()))
        file.write("\n\nMissing values per column:\n")
        file.write(str(missing_values))
        file.write("\n\nDuplicate rows:\n")
        file.write(str(duplicate_count))
        file.write("\n\nTarget distribution:\n")
        file.write(str(target_counts))
        file.write("\n\nTarget distribution percentage:\n")
        file.write(str(target_percentages.round(2)))

    print("\nStep 1 completed successfully.")
    print(f"Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()