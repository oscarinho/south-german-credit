# 💳 South German Credit Risk — CRISP-ML Project

A complete **CRISP-ML(Q)** (Cross-Industry Standard Process for Machine Learning with Quality Assurance) pipeline for predicting whether a credit applicant is a **good (0)** or **bad (1)** credit risk, on an imbalanced dataset where detecting the minority class is what matters most.

## Project Structure

```
south-german-credit/
├── app/
│   └── app.py              # Streamlit deployment app (real-time predictor)
├── data/
│   └── SouthGermanCredit.csv       # Dataset (1000 records, 20 features)
├── model/
│   ├── best_model.joblib           # Trained pipeline (preprocess + SMOTE + XGB)
│   └── model_metadata.pkl          # Metrics, params, feature columns
├── notebooks/
│   └── CRISP_ML_SouthGermanCredit.ipynb   # Full CRISP-ML analysis (Phases 1–6)
├── requirements.txt
└── README.md
```

## Dataset

**Source:** South German Credit (UCI Machine Learning Repository)
**Size:** 1,000 applicants, 20 features

### Key Features

| Category    | Features                                                          |
| ----------- | ----------------------------------------------------------------- |
| Account     | status, savings, credit_history, number_credits                   |
| Loan        | duration, amount, purpose, installment_rate, other_installment_plans |
| Applicant   | age, personal_status_sex, employment_duration, job, housing       |
| Other       | other_debtors, property, present_residence_since, people_liable, telephone, foreign_worker |

**Target:** `credit_risk` — 0 = good, 1 = bad (binary classification)
**Class Imbalance:** ~70% good vs ~30% bad (2.33 : 1)

---

## CRISP-ML(Q) Pipeline

### Phase 1 — Business Understanding

- **Goal:** Flag bad-credit applicants before a loan is approved
- **Business Value:** Avoid defaults (lost principal + interest), the costliest error
- **Cost asymmetry:** A False Negative (approve a bad credit) is far more expensive than a False Positive → **Recall on the bad class is the primary KPI**
- **Success Criteria:** Recall ≥ 0.50, ROC-AUC ≥ 0.75, train/test gap < 5%

---

### Phase 2 — Data Understanding

- 1,000 observations, no missing values
- Strong class imbalance (~30% bad credit)
- `duration` and `amount` right-skewed; `age` roughly normal
- Account `status` and `credit_history` separate risk strongly

---

### Phase 3 — Data Preparation

**Preprocessing (inside a `ColumnTransformer`):**

- `duration`, `amount` → Yeo-Johnson (PowerTransformer) + StandardScaler
- `age` → StandardScaler only
- Nominal categoricals → OneHotEncoder (`drop='first'`)
- Ordinal categoricals → OrdinalEncoder (preserves level order)

**Splitting:**

- 70/15/15 stratified train / validation / test (preserves the 70/30 ratio)

**Class Balancing:**

- RandomUnderSampler, SMOTE, KMeansSMOTE, **SMOTETomek** compared
- Resampling lives **inside** the imbalanced-learn pipeline → applied only on the training fold (no leakage)

---

### Phase 4 — Modeling

**Models Evaluated:**

- Logistic Regression (baseline)
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine
- Multi-Layer Perceptron
- XGBoost (tuned via GridSearchCV)

**Evaluation Strategy:**

- `RepeatedStratifiedKFold(5×3)` → every number is the mean over 15 folds
- GridSearchCV for XGBoost hyperparameters
- **XGBoost + SMOTETomek** selected for the best Recall / train-test stability balance

---

### Phase 5 — Evaluation

**Final model — stable 15-fold CV estimate** (reported instead of a single 150-row test draw, which is high-variance):

| Metric    | CV (mean ± sd) |
| --------- | -------------- |
| Accuracy  | 0.75 ± 0.04    |
| Precision | 0.60 ± 0.08    |
| Recall    | 0.50 ± 0.06    |
| F1 Score  | 0.54 ± 0.06    |
| ROC-AUC   | 0.77 ± 0.04    |

**Key Insights:**

- **Accuracy lies on imbalanced data** — without resampling, recall on the bad class falls to ~0.08
- **A single small test set is high-variance** — re-drawing the val/test split moves recall across 0.42–0.60; the original master's unseeded split reported 0.556, which sits at the ~82nd percentile of that distribution (a favorable but normal draw, not a better model)
- **Top drivers** (SHAP & permutation): account `status`, `credit_history`, `duration`, `amount`
- **The 0.50 threshold is wrong for credit** — with FN 5× costlier than FP, the cost-optimal threshold drops to ~0.06, lifting recall to ~0.87
- **Fairness** — subgroup gaps exist in `personal_status_sex` and `foreign_worker`; auditing is mandatory before deployment

---

### Phase 6 — Deployment

**Streamlit application** — real-time credit-risk scoring:

- Interactive applicant input form
- Good / bad prediction with class probabilities
- Selectable decision threshold (default 0.50 or cost-optimal for FN:FP = 5:1)
- Per-applicant SHAP risk factors (why this decision)
- Pipeline-safe inference (the saved model handles all preprocessing)

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Jupyter Notebook

```bash
jupyter lab notebooks/CRISP_ML_SouthGermanCredit.ipynb
```

### Run Streamlit App

```bash
streamlit run app/app.py
```

---

## Pipeline Architecture

```
Input Data → ColumnTransformer → SMOTETomek → XGBoost → Prediction
```

The complete pipeline is saved as a single object (`best_model.joblib`) that handles all preprocessing automatically during inference.

---

## Key Dependencies

- pandas, numpy — Data manipulation
- scikit-learn — ML pipeline and preprocessing
- imbalanced-learn — SMOTE / SMOTETomek for class balancing
- xgboost — Gradient boosting classifier
- shap — Model explainability
- matplotlib, seaborn — Visualization
- streamlit — Web application deployment
- joblib — Model serialization

---

*Built with CRISP-ML(Q) methodology and Streamlit*
