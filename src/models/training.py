import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

INPUT_DIR = "data/processed"

def main():
    X_train = pd.read_csv(f"{INPUT_DIR}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{INPUT_DIR}/y_train.csv").values.ravel()

    best_params = joblib.load("models/best_params.pkl")
    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    main()
