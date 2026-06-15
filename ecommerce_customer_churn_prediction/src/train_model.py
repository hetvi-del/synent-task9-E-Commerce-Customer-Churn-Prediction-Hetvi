import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "tenure",
    "warehousetohome",
    "hourspendonapp",
    "numberofdeviceregistered",
    "satisfactionscore",
    "numberofaddress",
    "orderamounthikefromlastyear",
    "couponused",
    "ordercount",
    "daysincelastorder",
    "cashbackamount",
    "orderfrequency",
    "customervaluescore",
    "highdistancecustomer",
    "complaintflag",
    "engagementscore",
]

CATEGORICAL_FEATURES = [
    "preferredlogindevice",
    "citytier",
    "preferredpaymentmode",
    "gender",
    "preferedordercat",
    "maritalstatus",
    "loyaltycategory",
    "recentcustomeractivity",
]

MODEL_MAP = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, random_state=42),
}


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_all_models(X_train, y_train):
    results = {}
    preprocessor = build_preprocessor()
    for model_name, estimator in MODEL_MAP.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        results[model_name] = pipeline
    return results


def evaluate_models(models, X_test, y_test):
    metrics = []
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
        )
    return pd.DataFrame(metrics).sort_values(by="roc_auc", ascending=False)


def select_best_model(models, metrics_df):
    best_model_name = metrics_df.iloc[0]["model"]
    return best_model_name, models[best_model_name]


def save_model(model_pipeline, filepath: str):
    joblib.dump(model_pipeline, filepath)


def run_training_pipeline(df: pd.DataFrame, model_path: str = "models/churn_model.pkl"):
    X_train, X_test, y_train, y_test = prepare_data(df)
    models = train_all_models(X_train, y_train)
    metrics_df = evaluate_models(models, X_test, y_test)
    best_model_name, best_pipeline = select_best_model(models, metrics_df)
    save_model(best_pipeline, model_path)
    return best_model_name, metrics_df, best_pipeline, X_test, y_test
