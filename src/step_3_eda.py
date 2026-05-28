import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def save_bar_chart(data, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(8, 5))
    data.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    # File paths
    input_path = Path("data/heart_disease_cleaned.csv")
    output_dir = Path("outputs/step_3_eda")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cleaned dataset
    df = pd.read_csv(input_path)

    print("\n==============================")
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("==============================")

    print(f"\nDataset shape: {df.shape[0]} rows and {df.shape[1]} columns")

    # Rename values for clearer charts
    df["target_label"] = df["target"].map({
        0: "No Heart Disease",
        1: "Heart Disease"
    })

    df["sex_label"] = df["sex"].map({
        0: "Female",
        1: "Male"
    })

    df["cp_label"] = df["cp"].map({
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic"
    })

    df["exang_label"] = df["exang"].map({
        0: "No",
        1: "Yes"
    })

    # 1. Target distribution
    target_counts = df["target_label"].value_counts()

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="target_label")
    plt.title("Heart Disease Distribution")
    plt.xlabel("Heart Disease Status")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(output_dir / "heart_disease_distribution.png")
    plt.close()

    # 2. Age distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="age", bins=15, kde=True)
    plt.title("Age Distribution of Patients")
    plt.xlabel("Age")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(output_dir / "age_distribution.png")
    plt.close()

    # 3. Heart disease by sex
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="sex_label", hue="target_label")
    plt.title("Heart Disease by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Number of Patients")
    plt.legend(title="Heart Disease Status")
    plt.tight_layout()
    plt.savefig(output_dir / "heart_disease_by_sex.png")
    plt.close()

    # 4. Heart disease by chest pain type
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="cp_label", hue="target_label")
    plt.title("Heart Disease by Chest Pain Type")
    plt.xlabel("Chest Pain Type")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=20)
    plt.legend(title="Heart Disease Status")
    plt.tight_layout()
    plt.savefig(output_dir / "heart_disease_by_chest_pain_type.png")
    plt.close()

    # 5. Cholesterol distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="chol", bins=15, kde=True)
    plt.title("Cholesterol Distribution")
    plt.xlabel("Serum Cholesterol")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(output_dir / "cholesterol_distribution.png")
    plt.close()

    # 6. Resting blood pressure distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="trestbps", bins=15, kde=True)
    plt.title("Resting Blood Pressure Distribution")
    plt.xlabel("Resting Blood Pressure")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(output_dir / "resting_blood_pressure_distribution.png")
    plt.close()

    # 7. Maximum heart rate distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="thalach", bins=15, kde=True)
    plt.title("Maximum Heart Rate Distribution")
    plt.xlabel("Maximum Heart Rate Achieved")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(output_dir / "maximum_heart_rate_distribution.png")
    plt.close()

    # 8. Exercise-induced angina vs heart disease
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="exang_label", hue="target_label")
    plt.title("Exercise-Induced Angina vs Heart Disease")
    plt.xlabel("Exercise-Induced Angina")
    plt.ylabel("Number of Patients")
    plt.legend(title="Heart Disease Status")
    plt.tight_layout()
    plt.savefig(output_dir / "exercise_angina_vs_heart_disease.png")
    plt.close()

    # 9. Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.drop(columns=["target_label", "sex_label", "cp_label", "exang_label"]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Heart Disease Features")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()

    # Save grouped summary tables
    target_counts.to_csv(output_dir / "target_distribution_table.csv")

    sex_summary = pd.crosstab(df["sex_label"], df["target_label"])
    sex_summary.to_csv(output_dir / "heart_disease_by_sex_table.csv")

    cp_summary = pd.crosstab(df["cp_label"], df["target_label"])
    cp_summary.to_csv(output_dir / "heart_disease_by_chest_pain_type_table.csv")

    exang_summary = pd.crosstab(df["exang_label"], df["target_label"])
    exang_summary.to_csv(output_dir / "exercise_angina_vs_heart_disease_table.csv")

    # Save EDA report
    with open(output_dir / "eda_report.txt", "w") as file:
        file.write("STEP 3: EXPLORATORY DATA ANALYSIS REPORT\n")
        file.write("========================================\n\n")
        file.write(f"Dataset used: {input_path}\n")
        file.write(f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns\n\n")

        file.write("Heart Disease Distribution:\n")
        file.write(str(target_counts))
        file.write("\n\n")

        file.write("Heart Disease by Sex:\n")
        file.write(str(sex_summary))
        file.write("\n\n")

        file.write("Heart Disease by Chest Pain Type:\n")
        file.write(str(cp_summary))
        file.write("\n\n")

        file.write("Exercise-Induced Angina vs Heart Disease:\n")
        file.write(str(exang_summary))
        file.write("\n\n")

        file.write("EDA charts saved in outputs/step_3_eda/.\n")

    print("\nStep 3 completed successfully.")
    print(f"EDA charts and reports saved in: {output_dir}")


if __name__ == "__main__":
    main()