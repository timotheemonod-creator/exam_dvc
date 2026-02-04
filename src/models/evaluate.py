import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

INPUT_DIR = "data/processed"

def main():
    X_test = pd.read_csv(f"{INPUT_DIR}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{INPUT_DIR}/y_test.csv").values.ravel()

    model = joblib.load("models/model.pkl")
    preds = model.predict(X_test)

    pd.DataFrame({"prediction": preds}).to_csv(f"{INPUT_DIR}/predictions.csv", index=False)

    metrics = {
        "mse": float(mean_squared_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }

    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()
