from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifact" / "final-model" / "model.pkl"
TARGET_COL = "etiqueta_fraude_simulada"
DEFAULT_DROP_COLS = [
    "id_siniestro",
    "id_poliza",
    "id_asegurado",
    "id_proveedor",
    "descripcion",
    "created_at",
    "updated_at",
]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    estimator: object
    param_distributions: dict


def load_model(model_path: Path | None = None):
    path = model_path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo final en {path}. "
            "Ejecuta el notebook 03 para exportarlo."
        )
    return joblib.load(path)


def predict(
    df: pd.DataFrame,
    model_path: Path | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    model = load_model(model_path)
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    return pd.DataFrame({"y_pred": preds, "y_prob": probs})


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    drop_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    columns_to_drop = list(drop_cols) if drop_cols is not None else list(
        DEFAULT_DROP_COLS
    )
    columns_to_drop.append(target_col)

    X = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
    datetime_cols = X.select_dtypes(include=[
        "datetime64[ns]",
        "datetime64[ns, UTC]",
    ]).columns
    if len(datetime_cols) > 0:
        X = X.drop(columns=list(datetime_cols))
    y = df[target_col].astype(int)
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, cat_cols, num_cols


def build_preprocess(
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, list(cat_cols)),
            ("num", numeric_pipeline, list(num_cols)),
        ]
    )


def split_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def default_model_configs(random_state: int = 42) -> list[ModelConfig]:
    return [
        ModelConfig(
            name="logistic_regression",
            estimator=LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
            param_distributions={
                "model__C": np.logspace(-3, 2, 10),
                "model__solver": ["lbfgs", "liblinear"],
            },
        ),
        ModelConfig(
            name="decision_tree",
            estimator=DecisionTreeClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            param_distributions={
                "model__max_depth": [3, 5, 8, 12, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        ModelConfig(
            name="random_forest",
            estimator=RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            param_distributions={
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [5, 10, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
    ]


class FraudModelTrainer:
    def __init__(
        self,
        preprocess,
        random_state: int = 42,
        n_iter: int = 20,
        cv: int = 5,
    ):
        self.preprocess = preprocess
        self.random_state = random_state
        self.n_iter = n_iter
        self.cv = cv

    def build_pipeline(self, estimator):
        return Pipeline(
            steps=[
                ("preprocess", self.preprocess),
                ("model", estimator),
            ]
        )

    def train(
        self,
        config: ModelConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        pipe = self.build_pipeline(config.estimator)
        search = RandomizedSearchCV(
            pipe,
            param_distributions=config.param_distributions,
            n_iter=self.n_iter,
            scoring="f1",
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_, search.best_score_


def train_models(
    trainer: FraudModelTrainer,
    configs: Sequence[ModelConfig],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    trained_models: dict[str, Pipeline] = {}
    results = []
    for config in configs:
        best_model, best_params, best_score = trainer.train(
            config,
            X_train,
            y_train,
        )
        trained_models[config.name] = best_model
        results.append(
            {
                "model": config.name,
                "best_score_cv_f1": best_score,
                "best_params": best_params,
            }
        )

    return trained_models, pd.DataFrame(results)


def export_models(
    trained_models: dict[str, Pipeline],
    artifact_dir: Path,
) -> list[Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for name, model in trained_models.items():
        model_path = artifact_dir / f"{name}.pkl"
        joblib.dump(model, model_path)
        saved_paths.append(model_path)
    return saved_paths


def build_predictions(
    trained_models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    pred_rows = []
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        pred_rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "y_true": y_test.values,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                }
            )
        )

    return pd.concat(pred_rows, ignore_index=True)
