import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed"

def main():
    X_train = pd.read_csv(f"{INPUT_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{INPUT_DIR}/X_test.csv")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
        f"{OUTPUT_DIR}/X_train_scaled.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
        f"{OUTPUT_DIR}/X_test_scaled.csv", index=False
    )

    joblib.dump(scaler, "models/scaler.pkl")

if __name__ == "__main__":
    main()
