"""Reproducible training script for the car price regression model.

Загружает датасет, делает feature engineering, обучает Ridge через GridSearchCV
с 10-кратной кросс-валидацией и сохраняет пайплайн в model_pipeline.pkl.

Usage:
    python train.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_BASE = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1"
TRAIN_URL = f"{DATA_BASE}/cars_train.csv"
TEST_URL = f"{DATA_BASE}/cars_test.csv"

CURRENT_YEAR = 2024
MODEL_PATH = Path(__file__).parent / "model_pipeline.pkl"
RANDOM_STATE = 42


def parse_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """`mileage`, `engine`, `max_power` приходят строками с единицами — извлекаем число."""
    df = df.copy()
    for col in ("mileage", "engine", "max_power"):
        df[col] = df[col].astype("string").str.extract(r"(\d+\.?\d*)")[0].astype(float)

    # torque: число в Nm + опциональный max_torque_rpm
    torque_str = df["torque"].astype("string")
    torque_num = torque_str.str.extract(r"(\d+\.?\d*)", expand=False).astype(float)
    is_kgm = torque_str.str.contains("kgm", case=False, na=False)
    torque_num = torque_num.where(~is_kgm, torque_num * 9.8)

    rpm_str = torque_str.str.replace(",", "", regex=False)
    df["max_torque_rpm"] = rpm_str.str.extract(r"(?:@|at)\s*(\d+)", expand=False).astype(float)
    df["torque"] = torque_num
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age"] = CURRENT_YEAR - df["year"]
    df["age_sq"] = df["age"] ** 2
    df["hp_per_liter"] = df["max_power"] / (df["engine"] / 1000.0)
    df["brand"] = df["name"].astype("string").str.split().str[0]
    return df


def build_pipeline(num_features: list[str], cat_features: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", Ridge())])


def main() -> None:
    print("→ Загружаем данные…")
    df_train = pd.read_csv(TRAIN_URL)
    df_test = pd.read_csv(TEST_URL)
    print(f"  train: {df_train.shape}, test: {df_test.shape}")

    df_train = df_train.drop_duplicates(
        subset=df_train.columns.difference(["selling_price"]), keep="first"
    ).reset_index(drop=True)

    df_train = parse_string_columns(df_train)
    df_test = parse_string_columns(df_test)

    num_cols = df_train.select_dtypes(include=["float", "int"]).columns
    medians = df_train[num_cols].median()
    df_train[num_cols] = df_train[num_cols].fillna(medians)
    df_test[num_cols] = df_test[num_cols].fillna(medians)

    df_train = add_engineered_features(df_train)
    df_test = add_engineered_features(df_test)

    drop_cols = ["selling_price", "name", "year"]
    X_train = df_train.drop(columns=drop_cols)
    y_train = df_train["selling_price"]
    X_test = df_test.drop(columns=drop_cols)
    y_test = df_test["selling_price"]

    num_features = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"  числовых: {len(num_features)}, категориальных: {len(cat_features)}")

    pipe = build_pipeline(num_features, cat_features)

    print("→ GridSearchCV (10-fold) по alpha…")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid={"model__alpha": np.logspace(-4, 4, 30)},
        cv=10,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print(f"  best alpha = {grid.best_params_['model__alpha']:.4f}")
    print(f"  CV R²     = {grid.best_score_:.4f}")

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)
    print(f"  train R²  = {r2_score(y_train, y_pred_train):.4f}")
    print(f"  test  R²  = {r2_score(y_test, y_pred_test):.4f}")
    print(f"  test  MSE = {mean_squared_error(y_test, y_pred_test):.3e}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best, f)
    print(f"→ Пайплайн сохранён в {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
