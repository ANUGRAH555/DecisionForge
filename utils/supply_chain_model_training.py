import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.supply_chain_preprocessing import preprocess_supply_chain_data


def train_supply_chain_models(df):
    """
    Train and compare Supply Chain regression models.
    Saves the best model and preprocessor.
    """

    # -------------------------------------------------
    # PREPROCESS DATA
    # -------------------------------------------------
    X_train, X_test, y_train, y_test, preprocessor = (
        preprocess_supply_chain_data(df)
    )

    # -------------------------------------------------
    # MODELS
    # -------------------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            objective="reg:squarederror"
        )
    }

    results = {}

    # -------------------------------------------------
    # TRAIN & EVALUATE
    # -------------------------------------------------
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mse ** 0.5,   # ✅ FIXED
            "R2": r2_score(y_test, y_pred)
        }

    # -------------------------------------------------
    # SELECT BEST MODEL (BY R2 SCORE)
    # -------------------------------------------------
    best_model_name = max(results, key=lambda x: results[x]["R2"])
    best_model = models[best_model_name]

    # -------------------------------------------------
    # SAVE MODEL & PREPROCESSOR
    # -------------------------------------------------
    joblib.dump(best_model, "models/supply_chain_model.pkl")
    joblib.dump(preprocessor, "models/supply_chain_preprocessor.pkl")

    return results, best_model_name
