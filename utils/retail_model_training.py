import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.retail_preprocessing import preprocess_retail_data


def train_retail_models(df: pd.DataFrame):
    """
    Train and evaluate Retail & E-Commerce models.
    Saves the best model and preprocessor.
    """

    # -------------------------------------------------
    # PREPROCESS DATA
    # -------------------------------------------------
    X_train, X_test, y_train, y_test, preprocessor = preprocess_retail_data(df)

    # Convert target to binary (important for XGBoost)
    y_train_bin = y_train.map({"No": 0, "Yes": 1})
    y_test_bin = y_test.map({"No": 0, "Yes": 1})

    # -------------------------------------------------
    # MODELS
    # -------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = {}

    # -------------------------------------------------
    # TRAIN & EVALUATE
    # -------------------------------------------------
    for name, model in models.items():
        model.fit(X_train, y_train_bin)

        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test_bin, y_pred),
            "precision": precision_score(y_test_bin, y_pred, zero_division=0),
            "recall": recall_score(y_test_bin, y_pred, zero_division=0),
            "f1_score": f1_score(y_test_bin, y_pred, zero_division=0)
        }

    # -------------------------------------------------
    # SELECT BEST MODEL (F1 SCORE)
    # -------------------------------------------------
    best_model_name = max(results, key=lambda x: results[x]["f1_score"])
    best_model = models[best_model_name]

    # -------------------------------------------------
    # SAVE MODEL & PREPROCESSOR
    # -------------------------------------------------
    joblib.dump(best_model, "models/retail_model.pkl")
    joblib.dump(preprocessor, "models/retail_preprocessor.pkl")

    return results, best_model_name
