import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier

from utils.banking_preprocessing import preprocess_banking_data


def train_banking_models(df):
    """
    Train and compare Banking Fraud Detection models
    including XGBoost.
    """

    # -------------------------------------------------
    # PREPROCESS DATA
    # -------------------------------------------------
    X_train, X_test, y_train, y_test, preprocessor = preprocess_banking_data(df)

    # Convert target to binary for XGBoost
    y_train_bin = y_train.map({"Yes": 1, "No": 0})
    y_test_bin = y_test.map({"Yes": 1, "No": 0})

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
        if name == "XGBoost":
            model.fit(X_train, y_train_bin)
            y_pred = model.predict(X_test)
            y_pred = ["Yes" if p == 1 else "No" for p in y_pred]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label="Yes"),
            "recall": recall_score(y_test, y_pred, pos_label="Yes"),
            "f1_score": f1_score(y_test, y_pred, pos_label="Yes")
        }

    # -------------------------------------------------
    # SELECT BEST MODEL (BY F1 SCORE)
    # -------------------------------------------------
    best_model_name = max(results, key=lambda x: results[x]["f1_score"])
    best_model = models[best_model_name]

    # -------------------------------------------------
    # SAVE MODEL & PREPROCESSOR
    # -------------------------------------------------
    joblib.dump(best_model, "models/banking_model.pkl")
    joblib.dump(preprocessor, "models/banking_preprocessor.pkl")

    return results, best_model_name
