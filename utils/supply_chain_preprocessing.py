import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocess_supply_chain_data(
    df: pd.DataFrame,
    target_column: str = "Sales",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Supply Chain Data Preprocessing Pipeline

    Dataset Columns:
    - ProductID
    - ProductCategory
    - WarehouseLocation
    - Supplier
    - LeadTime
    - DailyDemand
    - MonthlyDemand
    - CurrentStock
    - ReorderPoint
    - HoldingCost
    - ShortageCost
    - Sales (Target)

    Returns:
    X_train_processed, X_test_processed,
    y_train, y_test,
    fitted_preprocessor
    """

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    # -------------------------------------------------
    # FEATURES & TARGET
    # -------------------------------------------------
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # -------------------------------------------------
    # FEATURE GROUPS (LOCKED)
    # -------------------------------------------------
    categorical_features = [
        "ProductCategory",
        "WarehouseLocation",
        "Supplier"
    ]

    numerical_features = [
        "LeadTime",
        "DailyDemand",
        "MonthlyDemand",
        "CurrentStock",
        "ReorderPoint",
        "HoldingCost",
        "ShortageCost"
    ]

    # -------------------------------------------------
    # NUMERIC PIPELINE
    # -------------------------------------------------
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # -------------------------------------------------
    # CATEGORICAL PIPELINE
    # -------------------------------------------------
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # -------------------------------------------------
    # COLUMN TRANSFORMER
    # -------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    # -------------------------------------------------
    # TRAIN–TEST SPLIT
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # -------------------------------------------------
    # FIT & TRANSFORM
    # -------------------------------------------------
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )
