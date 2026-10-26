from __future__ import annotations

import json
import math
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


WORKBOOK = Path(
    r"E:\SDM_Research\Demirjian v. Kohler Data (AI Analyses - for Dr. Siddharth).xlsx"
)
OUTPUT = Path("assets/forensic-age-models.json")
OUTPUT_JS = Path("assets/forensic-age-models.js")
OUTPUT_VERSIONED_JS = Path("assets/forensic-age-models.v5.js")

TARGET = "Completed Age"
SEX_ENCODING = {"F": 0.0, "M": 1.0}
KOHLER_MIN_TEETH = 2
DROP_INCOMPLETE_KOHLER_TRAINING_ROWS = True
KOHLER_SOURCE_COLUMNS = ["18", "28", "38.1", "48"]

SEX_CONFIG = {
    "F": {
        "label": "Female",
        "train_sheet": "Female Data",
        "test_sheet": "Female Test Data",
        "total_column": "Total Score",
    },
    "M": {
        "label": "Male",
        "train_sheet": "Male Data",
        "test_sheet": "Male Test Data",
        "total_column": "Total",
    },
}

MODEL_BASES = {
    "sexSpecific": {
        "label": "Sex-specific",
        "description": "Models trained separately for male and female cases.",
    },
    "sexCombined": {
        "label": "Sex-combined",
        "description": "Models trained on male and female cases together with sex encoded as a predictor.",
    },
}

METHODS = {
    "method1a": {
        "label": "Method 1A",
        "shortLabel": "Demirjian total score of 8 teeth",
        "description": "Single continuous total score input.",
        "features": ["total_score"],
        "featureLabels": {"total_score": "Total score"},
        "sourceColumns": ["__total__"],
        "ranges": {"total_score": {"min": 0, "max": 100, "step": 0.01}},
    },
    "demirjian": {
        "label": "Method 1B",
        "shortLabel": "Demirjian method",
        "description": "Ordinal development stages for teeth 31 to 38.",
        "features": ["31", "32", "33", "34", "35", "36", "37", "38"],
        "featureLabels": {
            "31": "Tooth 31",
            "32": "Tooth 32",
            "33": "Tooth 33",
            "34": "Tooth 34",
            "35": "Tooth 35",
            "36": "Tooth 36",
            "37": "Tooth 37",
            "38": "Tooth 38",
        },
        "sourceColumns": ["31", "32", "33", "34", "35", "36", "37", "38"],
        "ranges": {
            "31": {"min": 0, "max": 9, "step": 1},
            "32": {"min": 0, "max": 9, "step": 1},
            "33": {"min": 0, "max": 9, "step": 1},
            "34": {"min": 0, "max": 9, "step": 1},
            "35": {"min": 0, "max": 9, "step": 1},
            "36": {"min": 0, "max": 9, "step": 1},
            "37": {"min": 0, "max": 9, "step": 1},
            "38": {"min": 0, "max": 9, "step": 1},
        },
    },
    "kohler": {
        "label": "Method 2",
        "shortLabel": "Kohler method (enter stages of at least two third molars)",
        "description": "Ordinal development stages for teeth 18, 28, 38, and 48.",
        "features": ["18", "28", "38", "48"],
        "featureLabels": {
            "18": "Tooth 18",
            "28": "Tooth 28",
            "38": "Tooth 38",
            "48": "Tooth 48",
        },
        "sourceColumns": ["18", "28", "38.1", "48"],
        "ranges": {
            "18": {"min": 1, "max": 10, "step": 1},
            "28": {"min": 1, "max": 10, "step": 1},
            "38": {"min": 1, "max": 10, "step": 1},
            "48": {"min": 1, "max": 10, "step": 1},
        },
    },
}

MODEL_LABELS = {
    "ols": "Ordinary Least Squares",
    "polynomial": "Polynomial Regression",
    "randomForest": "Random Forest Regression",
    "svr": "Support Vector Regression",
    "xgboost": "XGBoost Regression",
    "adaboost": "AdaBoost Regression",
}


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return value


def prepare_frame(df: pd.DataFrame, sex_code: str) -> pd.DataFrame:
    df = df.rename(columns={column: str(column).strip() for column in df.columns})
    total_column = SEX_CONFIG[sex_code]["total_column"]
    clean = df.copy()
    clean["total_score"] = clean[total_column]
    clean["sex_encoded"] = SEX_ENCODING[sex_code]
    return clean


def remove_incomplete_kohler_training_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not DROP_INCOMPLETE_KOHLER_TRAINING_ROWS:
        return df, []

    missing_mask = df[KOHLER_SOURCE_COLUMNS].isna().any(axis=1)
    removed: list[dict[str, Any]] = []
    for idx, row in df[missing_mask].iterrows():
        removed.append(
            {
                "excelRow": int(idx + 2),
                "sourceIndex": int(idx),
                "slNo": clean_float(row["Sl. No."]),
                "sex": row["Sex"],
                "completedAge": clean_float(row[TARGET]),
                "missingKohlerColumns": [
                    column for column in KOHLER_SOURCE_COLUMNS if pd.isna(row[column])
                ],
            }
        )
    return df.loc[~missing_mask].reset_index(drop=True), removed


def variant_key(features: list[str]) -> str:
    return "_".join(features)


def source_columns_for(method_key: str, features: list[str]) -> list[str]:
    method = METHODS[method_key]
    full_mapping = dict(zip(method["features"], method["sourceColumns"]))
    return [full_mapping[feature] for feature in features]


def kohler_feature_sets() -> list[list[str]]:
    features = list(METHODS["kohler"]["features"])
    return [
        list(combo)
        for size in range(KOHLER_MIN_TEETH, len(features) + 1)
        for combo in combinations(features, size)
    ]


def extract_xy(
    df: pd.DataFrame,
    method_key: str,
    include_sex_feature: bool = False,
    feature_subset: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    method = METHODS[method_key]
    features = list(feature_subset or method["features"])
    source_columns = (
        source_columns_for(method_key, features)
        if feature_subset is not None
        else list(method["sourceColumns"])
    )

    if include_sex_feature:
        features = ["sex_encoded", *features]
        source_columns = ["__sex__", *source_columns]

    assembled = pd.DataFrame(index=df.index)
    for feature, source_column in zip(features, source_columns):
        if source_column == "__total__":
            assembled[feature] = df["total_score"]
        elif source_column == "__sex__":
            assembled[feature] = df["sex_encoded"]
        else:
            assembled[feature] = df[source_column]

    assembled[TARGET] = df[TARGET]
    for column in [*features, TARGET]:
        assembled[column] = pd.to_numeric(assembled[column], errors="coerce")

    before = len(assembled)
    assembled = assembled.dropna(subset=[*features, TARGET])
    dropped = before - len(assembled)
    return assembled[features].to_numpy(dtype=float), assembled[TARGET].to_numpy(dtype=float), dropped


def tree_to_json(tree: DecisionTreeRegressor) -> dict[str, list[Any]]:
    inner = tree.tree_
    return {
        "childrenLeft": inner.children_left.tolist(),
        "childrenRight": inner.children_right.tolist(),
        "feature": inner.feature.tolist(),
        "threshold": [clean_float(x) for x in inner.threshold.tolist()],
        "value": [clean_float(x[0][0]) for x in inner.value.tolist()],
    }


def linear_to_json(model: LinearRegression) -> dict[str, Any]:
    return {
        "type": "linear",
        "intercept": clean_float(model.intercept_),
        "coefficients": [clean_float(x) for x in model.coef_.tolist()],
    }


def polynomial_to_json(model: Pipeline) -> dict[str, Any]:
    scaler: StandardScaler = model.named_steps["scale"]
    poly: PolynomialFeatures = model.named_steps["poly"]
    linear: LinearRegression = model.named_steps["linear"]
    return {
        "type": "polynomial",
        "degree": int(poly.degree),
        "mean": [clean_float(x) for x in scaler.mean_.tolist()],
        "scale": [clean_float(x) for x in scaler.scale_.tolist()],
        "powers": poly.powers_.astype(int).tolist(),
        "intercept": clean_float(linear.intercept_),
        "coefficients": [clean_float(x) for x in linear.coef_.tolist()],
    }


def random_forest_to_json(model: RandomForestRegressor) -> dict[str, Any]:
    return {
        "type": "randomForest",
        "trees": [tree_to_json(estimator) for estimator in model.estimators_],
    }


def svr_to_json(model: Pipeline) -> dict[str, Any]:
    scaler: StandardScaler = model.named_steps["scale"]
    svr: SVR = model.named_steps["svr"]
    return {
        "type": "svr",
        "kernel": svr.kernel,
        "gamma": clean_float(svr._gamma),
        "degree": int(svr.degree),
        "coef0": clean_float(svr.coef0),
        "intercept": clean_float(svr.intercept_[0]),
        "mean": [clean_float(x) for x in scaler.mean_.tolist()],
        "scale": [clean_float(x) for x in scaler.scale_.tolist()],
        "supportVectors": [[clean_float(x) for x in row] for row in svr.support_vectors_.tolist()],
        "dualCoef": [clean_float(x) for x in svr.dual_coef_[0].tolist()],
    }


def parse_xgb_base_score(model: XGBRegressor) -> float:
    config = json.loads(model.get_booster().save_config())
    raw = config["learner"]["learner_model_param"]["base_score"]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
    return float(raw)


def xgboost_to_json(model: XGBRegressor) -> dict[str, Any]:
    booster = model.get_booster()
    return {
        "type": "xgboost",
        "baseScore": clean_float(parse_xgb_base_score(model)),
        "trees": [json.loads(tree) for tree in booster.get_dump(dump_format="json")],
    }


def adaboost_to_json(model: AdaBoostRegressor) -> dict[str, Any]:
    estimator_count = len(model.estimators_)
    weights = model.estimator_weights_[:estimator_count]
    return {
        "type": "adaboost",
        "weights": [clean_float(x) for x in weights.tolist()],
        "trees": [tree_to_json(estimator) for estimator in model.estimators_],
    }


def metrics_for(model: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    return {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2_score(y_test, predictions)), 3),
    }


def model_suite(random_state: int) -> dict[str, Any]:
    return {
        "ols": LinearRegression(),
        "polynomial": Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("linear", LinearRegression()),
            ]
        ),
        "randomForest": RandomForestRegressor(
            n_estimators=50,
            max_depth=4,
            min_samples_leaf=3,
            random_state=random_state,
        ),
        "svr": Pipeline(
            [
                ("scale", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
            ]
        ),
        "xgboost": XGBRegressor(
            n_estimators=60,
            max_depth=2,
            learning_rate=0.08,
            subsample=0.95,
            colsample_bytree=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        ),
        "adaboost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=3),
            n_estimators=50,
            learning_rate=0.05,
            random_state=random_state,
        ),
    }


def train_model_artifact(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for model_key, model in model_suite(random_state).items():
        model.fit(x_train, y_train)
        models[model_key] = export_model(model_key, model)
        metrics[model_key] = metrics_for(model, x_test, y_test)
    return models, metrics


def export_model(key: str, model: Any) -> dict[str, Any]:
    if key == "ols":
        return linear_to_json(model)
    if key == "polynomial":
        return polynomial_to_json(model)
    if key == "randomForest":
        return random_forest_to_json(model)
    if key == "svr":
        return svr_to_json(model)
    if key == "xgboost":
        return xgboost_to_json(model)
    if key == "adaboost":
        return adaboost_to_json(model)
    raise ValueError(f"Unsupported model key: {key}")


def train() -> dict[str, Any]:
    if not WORKBOOK.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK}")

    artifact: dict[str, Any] = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceWorkbook": WORKBOOK.name,
        "target": TARGET,
        "sexEncoding": SEX_ENCODING,
        "modelBases": MODEL_BASES,
        "modelLabels": MODEL_LABELS,
        "sexes": {code: {"label": cfg["label"]} for code, cfg in SEX_CONFIG.items()},
        "trainingExclusions": {
            "dropIncompleteKohlerTrainingRows": DROP_INCOMPLETE_KOHLER_TRAINING_ROWS,
            "kohlerColumnsChecked": KOHLER_SOURCE_COLUMNS,
            "removedRows": {},
        },
        "methods": {
            key: {
                "label": method["label"],
                "shortLabel": method["shortLabel"],
                "description": method["description"],
                "features": method["features"],
                "featureLabels": method["featureLabels"],
                "ranges": method["ranges"],
                **(
                    {
                        "minimumRequiredFeatures": KOHLER_MIN_TEETH,
                        "partialFeatureSets": [
                            {"key": variant_key(features), "features": features}
                            for features in kohler_feature_sets()
                        ],
                    }
                    if key == "kohler"
                    else {}
                ),
            }
            for key, method in METHODS.items()
        },
        "trained": {
            "sexSpecific": {},
            "sexCombined": {},
        },
    }

    prepared: dict[str, dict[str, pd.DataFrame]] = {}
    for sex_index, (sex_code, sex_config) in enumerate(SEX_CONFIG.items()):
        raw_train_df = prepare_frame(
            pd.read_excel(WORKBOOK, sheet_name=sex_config["train_sheet"]), sex_code
        )
        train_df, removed_rows = remove_incomplete_kohler_training_rows(raw_train_df)
        artifact["trainingExclusions"]["removedRows"][sex_code] = removed_rows
        prepared[sex_code] = {
            "train": train_df,
            "test": prepare_frame(pd.read_excel(WORKBOOK, sheet_name=sex_config["test_sheet"]), sex_code),
        }
        train_df = prepared[sex_code]["train"]
        test_df = prepared[sex_code]["test"]
        artifact["trained"]["sexSpecific"][sex_code] = {}

        for method_index, method_key in enumerate(METHODS.keys()):
            x_train, y_train, dropped_train = extract_xy(train_df, method_key)
            x_test, y_test, dropped_test = extract_xy(test_df, method_key)
            random_state = 2026 + sex_index * 100 + method_index

            method_artifact: dict[str, Any] = {
                "trainRows": int(len(y_train)),
                "testRows": int(len(y_test)),
                "droppedTrainRows": int(dropped_train),
                "droppedTestRows": int(dropped_test),
                "models": {},
                "metrics": {},
            }

            method_artifact["models"], method_artifact["metrics"] = train_model_artifact(
                x_train, y_train, x_test, y_test, random_state
            )

            if method_key == "kohler":
                method_artifact["minimumRequiredFeatures"] = KOHLER_MIN_TEETH
                method_artifact["variants"] = {}
                for combo_index, features in enumerate(kohler_feature_sets()):
                    (
                        combo_x_train,
                        combo_y_train,
                        combo_dropped_train,
                    ) = extract_xy(train_df, method_key, feature_subset=features)
                    (
                        combo_x_test,
                        combo_y_test,
                        combo_dropped_test,
                    ) = extract_xy(test_df, method_key, feature_subset=features)
                    combo_models, combo_metrics = train_model_artifact(
                        combo_x_train,
                        combo_y_train,
                        combo_x_test,
                        combo_y_test,
                        random_state + 1000 + combo_index,
                    )
                    method_artifact["variants"][variant_key(features)] = {
                        "features": features,
                        "trainRows": int(len(combo_y_train)),
                        "testRows": int(len(combo_y_test)),
                        "droppedTrainRows": int(combo_dropped_train),
                        "droppedTestRows": int(combo_dropped_test),
                        "models": combo_models,
                        "metrics": combo_metrics,
                    }

            artifact["trained"]["sexSpecific"][sex_code][method_key] = method_artifact

    combined_train_df = pd.concat(
        [prepared[sex_code]["train"] for sex_code in SEX_CONFIG.keys()],
        ignore_index=True,
    )
    combined_test_df = pd.concat(
        [prepared[sex_code]["test"] for sex_code in SEX_CONFIG.keys()],
        ignore_index=True,
    )

    for method_index, method_key in enumerate(METHODS.keys()):
        x_train, y_train, dropped_train = extract_xy(
            combined_train_df, method_key, include_sex_feature=True
        )
        x_test, y_test, dropped_test = extract_xy(
            combined_test_df, method_key, include_sex_feature=True
        )
        random_state = 3026 + method_index

        method_artifact = {
            "trainRows": int(len(y_train)),
            "testRows": int(len(y_test)),
            "droppedTrainRows": int(dropped_train),
            "droppedTestRows": int(dropped_test),
            "includesSexFeature": True,
            "models": {},
            "metrics": {},
        }

        method_artifact["models"], method_artifact["metrics"] = train_model_artifact(
            x_train, y_train, x_test, y_test, random_state
        )

        if method_key == "kohler":
            method_artifact["minimumRequiredFeatures"] = KOHLER_MIN_TEETH
            method_artifact["variants"] = {}
            for combo_index, features in enumerate(kohler_feature_sets()):
                combo_x_train, combo_y_train, combo_dropped_train = extract_xy(
                    combined_train_df,
                    method_key,
                    include_sex_feature=True,
                    feature_subset=features,
                )
                combo_x_test, combo_y_test, combo_dropped_test = extract_xy(
                    combined_test_df,
                    method_key,
                    include_sex_feature=True,
                    feature_subset=features,
                )
                combo_models, combo_metrics = train_model_artifact(
                    combo_x_train,
                    combo_y_train,
                    combo_x_test,
                    combo_y_test,
                    random_state + 1000 + combo_index,
                )
                method_artifact["variants"][variant_key(features)] = {
                    "features": features,
                    "trainRows": int(len(combo_y_train)),
                    "testRows": int(len(combo_y_test)),
                    "droppedTrainRows": int(combo_dropped_train),
                    "droppedTestRows": int(combo_dropped_test),
                    "includesSexFeature": True,
                    "models": combo_models,
                    "metrics": combo_metrics,
                }

        artifact["trained"]["sexCombined"][method_key] = method_artifact

    return artifact


def main() -> None:
    artifact = train()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, separators=(",", ":"))
    OUTPUT.write_text(serialized, encoding="utf-8")
    OUTPUT_JS.write_text(f"window.FORENSIC_AGE_MODELS={serialized};\n", encoding="utf-8")
    OUTPUT_VERSIONED_JS.write_text(
        f"window.FORENSIC_AGE_MODELS={serialized};\n", encoding="utf-8"
    )

    summary: dict[str, Any] = {"sexSpecific": {}, "sexCombined": {"Combined": {}}}
    for sex, methods in artifact["trained"]["sexSpecific"].items():
        summary["sexSpecific"][sex] = {
            method: {
                "trainRows": values["trainRows"],
                "testRows": values["testRows"],
                "metrics": values["metrics"],
            }
            for method, values in methods.items()
        }
    summary["sexCombined"]["Combined"] = {
        method: {
            "trainRows": values["trainRows"],
            "testRows": values["testRows"],
            "metrics": values["metrics"],
        }
        for method, values in artifact["trained"]["sexCombined"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KB)")
    print(f"Saved {OUTPUT_JS} ({OUTPUT_JS.stat().st_size / 1024:.1f} KB)")
    print(f"Saved {OUTPUT_VERSIONED_JS} ({OUTPUT_VERSIONED_JS.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
