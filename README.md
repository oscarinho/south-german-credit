# South German Credit Risk Prediction

A machine learning project for classifying credit applicants as good or bad risk using the South German Credit dataset.

## Overview

This project addresses the challenge of credit risk assessment with imbalanced data (70% good credit, 30% bad credit). It includes exploratory analysis, model comparison with multiple classifiers, class imbalance handling techniques, and a Streamlit web application for real-time predictions.

**Best Model**: XGBoost with SMOTETomek resampling

Metrics are reported as a **stable 15-fold cross-validated estimate** (mean ± sd) of the final pipeline, rather than a single 150-row test draw — a small held-out test set is high-variance on this dataset (see notebook §5.3b).

| Metric    | CV (mean ± sd) |
|-----------|----------------|
| Accuracy  | 0.75 ± 0.04    |
| Recall    | 0.50 ± 0.06    |
| F1 Score  | 0.54 ± 0.06    |
| ROC-AUC   | 0.77 ± 0.04    |

> The original master's notebook reported Recall ≈ 0.556 from an *unseeded* val/test split; that value sits at ~the 82nd percentile of the split-to-split distribution — a favorable but normal draw of the same process, not a different model.

## Project Structure

```
south_german_credit/
├── app/
│   ├── app.py                  # Streamlit web application
│   ├── config.py               # Paths, feature groups, hyperparameters
│   ├── data_loader.py          # Loading + 70/15/15 stratified split
│   ├── preprocessing.py        # ColumnTransformer pipelines
│   ├── models.py               # 7 algorithms + tuned XGBoost
│   ├── training.py             # CV, resamplers, GridSearch helpers
│   └── evaluation.py           # Metrics, confusion matrix, ROC, boxplots
├── archive/                    # Original master's exercise + reference paper
├── data/
│   └── SouthGermanCredit.csv   # Dataset (1000 records, 21 features)
├── models/
│   └── best_model.joblib       # Trained XGBoost + SMOTETomek pipeline
├── notebooks/
│   └── CRISP_ML_SouthGermanCredit.ipynb        # Main analysis notebook
├── requirements.txt
└── README.md
```

## Dataset

The South German Credit dataset contains 1000 credit applicants with 20 features:

| Feature | Description |
|---------|-------------|
| `status` | Account status (1-4) |
| `duration` | Loan duration in months |
| `credit_history` | Payment history (0-4) |
| `purpose` | Loan purpose (0-10) |
| `amount` | Credit amount in DM |
| `savings` | Savings account balance (1-5) |
| `employment_duration` | Current employment length (1-5) |
| `installment_rate` | Installment as % of income (1-4) |
| `personal_status_sex` | Personal status and sex (1-4) |
| `other_debtors` | Other debtors/guarantors (1-3) |
| `present_residence_since` | Years at current residence (1-4) |
| `property` | Property type (1-4) |
| `age` | Age in years |
| `other_installment_plans` | Other installment plans (1-3) |
| `housing` | Housing type (1-3) |
| `number_credits` | Number of existing credits (1-4) |
| `job` | Job type (1-4) |
| `people_liable` | Number of dependents (1-2) |
| `telephone` | Has telephone (1-2) |
| `foreign_worker` | Is foreign worker (1-2) |

**Target**: `credit_risk` (0 = bad, 1 = good)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd south_german_credit

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Web Application

```bash
# 1) Precompute the dashboard artifacts (one time, ~1 min) — tables + figures
python -m app.precompute

# 2) Launch the multi-page CRISP-ML dashboard + predictor
streamlit run app/app.py
```

The app is a **multi-page CRISP-ML dashboard**: Resumen, Datos (EDA), Modelado,
Evaluación (CV estable + varianza del split), Fairness, and a live **Predictor**
with a cost-optimal decision threshold (FN:FP = 5:1) and per-applicant SHAP
explanations. Heavy results are precomputed into `app/artifacts/` and loaded
instantly; the predictor runs live against `models/best_model.joblib`.

### Run the Notebooks

Open the notebooks in Jupyter or Google Colab:

```bash
jupyter notebook notebooks/CRISP_ML_SouthGermanCredit.ipynb
```

## Methodology

### 1. Data Preprocessing
- Column renaming (German to English)
- Train/Validation/Test split (70/15/15)
- Feature scaling with StandardScaler

### 2. Models Evaluated
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine
- Multi-Layer Perceptron
- XGBoost

### 3. Class Imbalance Handling
- Random Undersampling
- SMOTE (Synthetic Minority Oversampling)
- KMeans-SMOTE
- SMOTETomek (hybrid approach)

### 4. Evaluation
- Cross-validation with RepeatedStratifiedKFold
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Focus on Recall to minimize false negatives (missed bad credits)

## Key Findings

1. **Class Imbalance Impact**: Without resampling, models achieved high accuracy but poor recall for the minority class (bad credit)

2. **SMOTETomek Effectiveness**: Combining SMOTE oversampling with Tomek links cleaning significantly improved recall from ~0.18 to ~0.53

3. **Best Configuration**: XGBoost + SMOTETomek provides the best balance between detecting bad credits (recall) and overall performance (F1)

## Tech Stack

- **Python** 3.8+
- **pandas** / **numpy** - Data manipulation
- **scikit-learn** - ML pipelines and models
- **XGBoost** - Gradient boosting classifier
- **imbalanced-learn** - Resampling techniques
- **Streamlit** - Web application
- **matplotlib** / **seaborn** - Visualization

## License

This project is for educational purposes as part of the "Inteligencia Artificial y Aprendizaje Automático" course at Tecnológico de Monterrey.
