# ML Placement Prediction

A Flask dashboard for a 50,000-student placement dataset.

## Included modules

- Data loading and schema inspection
- Missing-value handling
- Exploratory Data Analysis with plots and statistics
- Simple Linear Regression
- Linear Regression with Gradient Descent
- Logistic Regression
- Multi-model Salary Regression: Linear, Ridge, Random Forest
- Classification Model Training: Logistic Regression, Decision Tree, Random Forest, KNN
- Model comparison with accuracy, precision, recall, F1 and confusion matrices
- Saved trained models under `models/trained/`
- Interactive student placement prediction

## Run in PyCharm

Use Python **3.13** (recommended for a clean environment).

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

### If you already have an old `.venv`

The old archive contained a Python 3.14 environment. Do not reuse that environment. Delete `.venv`, create a fresh Python 3.13 environment, and install from `requirements.txt`.

## Important fixes made

- Corrected the dataset filename mismatch (`50K` vs `50k`).
- Corrected `AttendancePercent` and `Internships` feature names.
- Removed import-time model execution.
- Made missing-value processing safe for 50,000 rows; exact KNN imputation is used for smaller datasets and a memory-safe median fallback is used for this large dataset.
- Added leakage-safe preprocessing for classification.
- Added model training and prediction pages.
- Added saved model artifacts and metrics.
- Added a proper `requirements.txt`.
