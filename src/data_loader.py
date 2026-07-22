import os
import pandas as pd


def load_and_clean_data(file_path):
    """
    Load and clean Telco Customer Churn dataset.
    """

    # Load dataset
    df = pd.read_csv(file_path)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Remove customer ID
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    return df


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "Telco-Customer-Churn.csv"
    )

    print("=" * 50)
    print("Loading Dataset...")
    print("=" * 50)

    df = load_and_clean_data(DATA_PATH)

    # Dataset overview
    print(f"Dataset shape: {df.shape}")

    print("\nDataset information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print(f"\nDuplicated rows: {df.duplicated().sum()}")

    # Save cleaned dataset
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(
        processed_dir,
        "cleaned_data.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to:")
    print(output_path)