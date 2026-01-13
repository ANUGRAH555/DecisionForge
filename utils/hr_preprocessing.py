import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocess_hr_data(
    df: pd.DataFrame,
    target_column: str = "Attrition",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    HR Data Preprocessing Pipeline

    Steps:
    1. Handle missing values
    2. Encode categorical variables
    3. Scale numerical features
    4. Split train/test data

    Returns:
    X_train_processed, X_test_processed,
    y_train, y_test,
    fitted_preprocessor
    """

    # -----------------------------
    # Validate input
    # -----------------------------
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    # -----------------------------
    # Separate features & target
    # -----------------------------
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # -----------------------------
    # Define feature groups
    # -----------------------------
    categorical_features = [
        "Gender",
        "Department",
        "JobRole",
        "OverTime"
    ]

    numerical_features = [
        "Age",
        "MonthlyIncome",
        "JobSatisfaction",
        "YearsAtCompany"
    ]

    # -----------------------------
    # Numerical pipeline
    # -----------------------------
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # -----------------------------
    # Categorical pipeline
    # -----------------------------
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # -----------------------------
    # Combine pipelines
    # -----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    # -----------------------------
    # Train–test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # -----------------------------
    # Fit on train, transform both
    # -----------------------------
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )
