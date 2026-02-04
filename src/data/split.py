import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/raw/raw.csv"
OUTPUT_DIR = "data/processed"

def main():
    df = pd.read_csv(INPUT_PATH)

    y = df["silica_concentrate"]
    X = df.drop(columns=["silica_concentrate"])

    # garder uniquement les colonnes numériques
    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

if __name__ == "__main__":
    main()
