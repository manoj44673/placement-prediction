from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "trained"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PlacementStatus"
# Deliberately exclude IDs, target/leakage columns, and the salary outcome.
FEATURES = [
    "Gender", "City", "CollegeTier", "Stream", "Specialisation", "Hostel",
    "HistoryOfBacklogs", "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
    "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8", "CGPA",
    "AttendancePercent", "Internships", "Projects", "Workshops", "Certifications",
    "Publications", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore",
    "MockInterviewScore", "ExtraCurricular", "CGPA_Tier"
]

NUMERIC_FEATURES = [
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4", "SGPA_Sem5",
    "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8", "CGPA", "AttendancePercent",
    "Internships", "Projects", "Workshops", "Certifications", "Publications",
    "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore",
    "ExtraCurricular"
]
CATEGORICAL_FEATURES = [c for c in FEATURES if c not in NUMERIC_FEATURES]


def _make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def _prepare(df):
    data = df.copy()
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")
    X = data[FEATURES].copy()
    y = pd.to_numeric(data[TARGET], errors="coerce")
    valid = y.notna() & y.isin([0, 1])
    return X.loc[valid], y.loc[valid].astype(int)


def train_models(df, force=False):
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists() and not force:
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    X, y = _prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = _make_preprocessor()
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    results = []
    best_name = None
    best_score = -1

    for name, estimator in estimators.items():
        pipeline = Pipeline([
            ("preprocessor", base),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        result = {
            "model": name,
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        results.append(result)

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, MODEL_DIR / filename)

        if accuracy > best_score:
            best_score = accuracy
            best_name = name

    # Keep the best model under a stable name for prediction.
    best_file = best_name.lower().replace(" ", "_") + ".joblib"
    stable = MODEL_DIR / "best_model.joblib"
    joblib.dump(joblib.load(MODEL_DIR / best_file), stable)

    payload = {
        "target": TARGET,
        "features": FEATURES,
        "samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "models": results,
        "best_model": best_name,
        "best_accuracy": round(float(best_score), 4),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_best_model():
    path = MODEL_DIR / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError("No trained model found. Train the models first.")
    return joblib.load(path)


def predict_student(values):
    df = pd.DataFrame([values])
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    df = df[FEATURES]
    model = load_best_model()
    prediction = int(model.predict(df)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])
    return {
        "prediction": prediction,
        "label": "Placed" if prediction == 1 else "Not Placed",
        "probability": round(probability * 100, 2) if probability is not None else None,
    }


def dataset_options(df):
    options = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            options[col] = sorted(df[col].dropna().astype(str).unique().tolist())
    return options
