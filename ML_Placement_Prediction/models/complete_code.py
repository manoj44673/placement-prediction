"""Standalone sanity check for the Placement Prediction project."""
from pathlib import Path
import pandas as pd
from models.load import load_data
from models.feature_engg import knn_impute, run_regression_models
from models.classification import train_models

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    df = load_data()
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values before cleaning:", int(df.isna().sum().sum()))
    cleaned = knn_impute(df)["df"]
    print("Missing values after KNN cleaning:", int(cleaned.isna().sum().sum()))
    print("\nRegression:")
    print(run_regression_models(df))
    print("\nClassification:")
    print(train_models(df, force=True))

if __name__ == "__main__":
    main()
