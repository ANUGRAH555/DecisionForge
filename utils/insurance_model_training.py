import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.insurance_preprocessing import preprocess_insurance_data


def train_insurance_models(df: pd.DataFrame):
    """
    Train and evaluate Insurance Fraud models.
    Saves the best model and preprocessor.
    """

    # -------------------------------------------------
    # PREPROCESS DATA
    # -------------------------------------------------
    X_train, X_test, y_train, y_test, preprocessor = preprocess_insurance_data(df)

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
        )
    }

    results = {}

    # -------------------------------------------------
    # TRAIN & EVALUATE
    # -------------------------------------------------
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label="Yes", zero_division=0),
            "recall": recall_score(y_test, y_pred, pos_label="Yes", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, pos_label="Yes", zero_division=0)
        }

    # -------------------------------------------------
    # SELECT BEST MODEL (BY F1 SCORE)
    # -------------------------------------------------
    best_model_name = max(results, key=lambda x: results[x]["f1_score"])
    best_model = models[best_model_name]

    # -------------------------------------------------
    # SAVE MODEL & PREPROCESSOR
    # -------------------------------------------------
    joblib.dump(best_model, "models/insurance_model.pkl")
    joblib.dump(preprocessor, "models/insurance_preprocessor.pkl")

    return results, best_model_name
