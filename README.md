# Diabetes Prediction using Machine Learning

A binary classifier that predicts whether a patient is diabetic from diagnostic measurements, built with a linear Support Vector Machine on the PIMA Indians Diabetes dataset.

## Overview

This is an educational machine-learning project covering the full workflow: exploratory data analysis, data-quality checks, leak-free preprocessing, model training, evaluation and single-record prediction. Everything is in one notebook: [`notebooks/diabetes_prediction.ipynb`](notebooks/diabetes_prediction.ipynb).

## Problem Statement

Given a patient's diagnostic measurements (glucose, BMI, age, etc.), predict whether the patient has diabetes (`Outcome = 1`) or not (`Outcome = 0`).

## Objectives

- Explore the dataset and check its data quality.
- Preprocess the data without data leakage.
- Train a linear SVM classifier.
- Evaluate it with accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC.
- Predict the class for a new patient record.

## Dataset

- **Name:** PIMA Indians Diabetes dataset (originally from the National Institute of Diabetes and Digestive and Kidney Diseases; publicly available, for example on Kaggle and the UCI ML repository).
- **File:** `data/diabetes.csv`
- **Size:** 768 records, 8 input features, 1 target.
- **Class balance:** 500 Non-Diabetic (65.1%) and 268 Diabetic (34.9%).
- **Population:** female patients of PIMA Indian heritage, aged 21 or older.

## Features

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin-fold thickness (mm) |
| `Insulin` | Serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function (family-history score) |
| `Age` | Age in years |

**Target:** `Outcome` (0 = Non-Diabetic, 1 = Diabetic).

## Methodology

1. Load the data and inspect it.
2. Explore class balance, feature distributions and correlations.
3. Check data quality (zeros that represent missing measurements).
4. Split into train and test sets (80/20, stratified, `random_state=2`).
5. Fit the imputer and scaler on the training set only, then transform train and test.
6. Train the linear SVM (and a Logistic Regression baseline) on the training set.
7. Evaluate on the untouched test set and predict on a new record.

## Data Preprocessing

- **Invalid zeros:** `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` contain zeros that are physiologically impossible (5, 35, 227, 374 and 11 rows respectively), so they are treated as missing values. `Pregnancies = 0` is a valid value and is left unchanged.
- **Imputation:** the missing values are filled with the **median of the training set**.
- **Scaling:** `StandardScaler`, fitted on the training set only.
- **No data leakage:** the split happens before any statistic is learned, and the test set and new prediction inputs only use the already-fitted imputer and scaler. Column names are kept throughout, so the prediction input has the same feature names and order as the training data.

## Exploratory Data Analysis

The notebook contains a few purposeful plots:

- Class distribution of `Outcome`.
- Feature distributions split by class (which also reveal the spikes at zero).
- Share of invalid zeros per feature.
- Correlation heatmap (invalid zeros treated as missing).

`Glucose` is the feature most strongly related to `Outcome` (correlation about 0.49).

## Model

**Linear SVM:** `SVC(kernel="linear")`. A Logistic Regression model is trained on the same preprocessed data as a simple baseline for comparison.

## Evaluation

Metrics are computed on a held-out test set of 154 patients (100 Non-Diabetic, 54 Diabetic). Precision, recall and F1 refer to the Diabetic class; ROC-AUC uses the SVM decision scores.

## Results

Linear SVM (main model):

| Set | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Train | 0.7785 | 0.7349 | 0.5701 | 0.6421 | 0.8496 |
| Test | 0.7727 | 0.7568 | 0.5185 | 0.6154 | 0.8200 |

Test-set confusion matrix (rows = true, columns = predicted):

| | Predicted Non-Diabetic | Predicted Diabetic |
|---|---|---|
| **True Non-Diabetic** | 91 | 9 |
| **True Diabetic** | 26 | 28 |

Baseline comparison on the test set:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Linear SVM | 0.7727 | 0.7568 | 0.5185 | 0.6154 | 0.8200 |
| Logistic Regression | 0.7403 | 0.6842 | 0.4815 | 0.5652 | 0.8141 |

The model is better than always predicting the majority class (about 65% accuracy), but it misses roughly half of the diabetic patients in the test set. The original version of this project scaled the full dataset before splitting; after correcting this, the test accuracy is unchanged, while train accuracy and ROC-AUC differ slightly.

## Example Prediction

```python
sample_patient = {
    "Pregnancies": 10, "Glucose": 139, "BloodPressure": 80, "SkinThickness": 0,
    "Insulin": 0, "BMI": 27.1, "DiabetesPedigreeFunction": 1.441, "Age": 57,
}
label, name = predict_patient(sample_patient)
# Predicted class: Diabetic (1)
```

`SkinThickness = 0` and `Insulin = 0` are treated as missing and imputed with the training medians, exactly as in training. The output is a model prediction, not a diagnosis.

## Technologies Used

Python, NumPy, pandas, Matplotlib, seaborn, scikit-learn, Jupyter Notebook.

## Project Structure

```
diabetes-prediction-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── diabetes.csv
└── notebooks/
    └── diabetes_prediction.ipynb
```

## How to Run

```bash
git clone https://github.com/<your-username>/diabetes-prediction-ml.git
cd diabetes-prediction-ml
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
jupyter notebook notebooks/diabetes_prediction.ipynb
```

Then run all cells. The notebook was executed with Python 3.12, scikit-learn 1.8, pandas 3.0, NumPy 2.4, Matplotlib 3.10 and seaborn 0.13.

## Limitations

- Small dataset (768 records); the test set has only 154 patients, so metrics vary with the split.
- The dataset covers only female patients of PIMA Indian heritage aged 21+, so the model may not generalise to other groups.
- About 49% of `Insulin` and 30% of `SkinThickness` values are missing, and median imputation is only a simple approximation.
- The classes are imbalanced and recall for the Diabetic class is limited (about 52% on the test set).
- Results come from a single train-test split, without cross-validation or hyperparameter tuning.
- The model was not validated on clinical data.

## Future Improvements

- Stratified k-fold cross-validation for more reliable estimates.
- Hyperparameter tuning, class weighting or threshold tuning to improve recall.
- Additional models such as Random Forest or Gradient Boosting.
- More advanced imputation (KNN or iterative imputation).
- Model explainability (coefficients / feature importance).

## Disclaimer

This project is an educational machine-learning demonstration and is not a medically validated diagnostic tool. It must not be used for medical diagnosis or treatment decisions.
