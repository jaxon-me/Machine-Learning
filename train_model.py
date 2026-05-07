from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "Target Pressure (bar)"
ID_COLUMN = "ID"


def create_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def drop_columns_if_present(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return frame
    return frame.drop(columns=existing)


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in features.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", create_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(random_state=42)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BLEVE target pressure model.")
    parser.add_argument("--train", type=Path, default=Path("train.csv"), help="Path to training CSV.")
    parser.add_argument("--test", type=Path, default=Path("test.csv"), help="Path to test CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Path to write predictions CSV.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("model.joblib"),
        help="Path to write trained model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.train
    test_path = args.test

    train_df = pd.read_csv(train_path)
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Training data missing target column: {TARGET_COLUMN}")

    train_df = train_df.dropna(subset=[TARGET_COLUMN])
    if train_df.empty:
        raise ValueError("Training data has no rows with a target value.")

    features = drop_columns_if_present(train_df, [TARGET_COLUMN, ID_COLUMN])
    target = train_df[TARGET_COLUMN]

    pipeline = build_pipeline(features)

    try:
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=5,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=-1,
        )
        rmse = -scores["test_rmse"].mean()
        r2 = scores["test_r2"].mean()
    except ValueError:
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=5,
            scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
            n_jobs=-1,
        )
        rmse = (-scores["test_mse"].mean()) ** 0.5
        r2 = scores["test_r2"].mean()
    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation R2: {r2:.6f}")

    pipeline.fit(features, target)
    joblib.dump(pipeline, args.model_out)
    print(f"Saved model to {args.model_out.resolve()}")

    test_df = pd.read_csv(test_path)
    test_features = drop_columns_if_present(test_df, [TARGET_COLUMN, ID_COLUMN])
    missing_columns = set(features.columns) - set(test_features.columns)
    extra_columns = set(test_features.columns) - set(features.columns)
    if missing_columns or extra_columns:
        raise ValueError(
            "Test data columns do not match training features. "
            f"Missing: {sorted(missing_columns)}. Extra: {sorted(extra_columns)}."
        )
    test_features = test_features[features.columns]

    test_predictions = pipeline.predict(test_features)

    if ID_COLUMN in test_df.columns:
        output_df = pd.DataFrame({ID_COLUMN: test_df[ID_COLUMN], TARGET_COLUMN: test_predictions})
    else:
        output_df = pd.DataFrame({TARGET_COLUMN: test_predictions})
    output_df.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
