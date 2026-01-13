import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.customer_preprocessing import preprocess_customer_data


def train_customer_models(df):
    """
    Train and compare Customer Churn models.
    Saves best model & preprocessor.
    """

    # -------------------------------------------------
    # PREPROCESS DATA (ONLY SOURCE OF X & y)
    # -------------------------------------------------
    X_train, X_test, y_train, y_test, preprocessor = preprocess_customer_data(df)

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
            max_depth=5,
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
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }

    # -------------------------------------------------
    # SELECT BEST MODEL (F1 SCORE)
    # -------------------------------------------------
    best_model_name = max(results, key=lambda x: results[x]["f1_score"])
    best_model = models[best_model_name]

    # -------------------------------------------------
    # SAVE ARTIFACTS
    # -------------------------------------------------
    joblib.dump(best_model, "models/customer_model.pkl")
    joblib.dump(preprocessor, "models/customer_preprocessor.pkl")

    return results, best_model_name
