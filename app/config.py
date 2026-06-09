"""Configuration: paths, feature groups, hyperparameters."""
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "SouthGermanCredit.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

# Reproducibility
RANDOM_STATE = 42

# Split sizes
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# German -> English column translation (from codetable.txt)
COLUMN_TRANSLATION = {
    "laufkont": "status",
    "laufzeit": "duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence_since",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}

# Feature groups
NUMERIC_FEATURES = ["duration", "amount"]          # PowerTransform + Scaler
NUMERIC_FEATURES_AGE = ["age"]                      # Scaler only
CATEGORICAL_FEATURES = [
    "status", "credit_history", "purpose", "savings",
    "personal_status_sex", "other_debtors", "other_installment_plans",
    "housing", "telephone", "foreign_worker",
]
ORDINAL_FEATURES = [
    "employment_duration", "installment_rate",
    "present_residence_since", "property", "number_credits", "job",
]
TARGET = "credit_risk"

# Best hyperparameters from GridSearchCV in the original master's analysis
BEST_MODEL_PARAMS = {
    "alpha": 0,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "reg_lambda": 1,
    "learning_rate": 0.2,
    "max_depth": 4,
    "n_estimators": 200,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "n_jobs": -1,
}

# Cross-validation
CV_SPLITS = 5
CV_REPEATS = 3
SCORING_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
