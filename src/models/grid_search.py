import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

INPUT_DIR = "data/processed"

def main():
    X_train = pd.read_csv(f"{INPUT_DIR}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{INPUT_DIR}/y_train.csv").values.ravel()

    model = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring="neg_mean_squared_error")
    gs.fit(X_train, y_train)

    joblib.dump(gs.best_params_, "models/best_params.pkl")

if __name__ == "__main__":
    main()
