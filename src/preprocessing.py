import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, target_col='Churn',
                    test_size=0.2,
                    random_state=42):
    """
    Preprocess dataset:
    - Encode target
    - One-Hot Encoding
    - Train/Test Split
    - Standard Scaling
    """

    # Copy dữ liệu
    data = df.copy()

    # Encode target
    data[target_col] = data[target_col].map({"Yes": 1, "No": 0})

    # Feature / Target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Numerical & Categorical columns
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # One-Hot Encoding
    X = pd.get_dummies(
        X,
        columns=cat_cols,
        drop_first=True,
        dtype=int
    )

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Standardization
    scaler = StandardScaler()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(BASE_DIR, "src"))

    from data_loader import load_and_clean_data

    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "Telco-Customer-Churn.csv"
    )

    df = load_and_clean_data(DATA_PATH)

    print("=" * 50)
    print("Dataset Information")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    print(f"\nDuplicated rows: {df.duplicated().sum()}")

    # Save cleaned dataset
    processed_path = os.path.join(
        BASE_DIR,
        "data",
        "processed"
    )

    os.makedirs(processed_path, exist_ok=True)

    cleaned_file = os.path.join(
        processed_path,
        "cleaned_data.csv"
    )

    df.to_csv(cleaned_file, index=False)

    print(f"\nCleaned data saved to:")
    print(cleaned_file)

    # Preprocessing
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("\nTrain/Test Split")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")

    print("\nPreprocessing completed successfully!")