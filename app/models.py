"""Model definitions used in the comparison."""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from app.config import RANDOM_STATE, BEST_MODEL_PARAMS


def _xgb():
    """Lazy import so the package works even if xgboost isn't installed yet."""
    from xgboost import XGBClassifier
    return XGBClassifier


def get_all_models():
    """Return a dict of {name: estimator} with sensible defaults (matches
    the original master's analysis)."""
    models = {
        "LogisticRegression": LogisticRegression(
            C=1, penalty="l2", solver="liblinear",
            max_iter=1000, random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            metric="manhattan", n_neighbors=21, weights="uniform",
        ),
        "DecisionTree": DecisionTreeClassifier(
            criterion="entropy", max_depth=4,
            min_samples_leaf=5, min_samples_split=2,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=4,
            max_features="sqrt", min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "SVM": SVC(
            C=1.0, kernel="rbf", gamma="scale",
            probability=True, random_state=RANDOM_STATE,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(4,), activation="logistic",
            alpha=1e-4, max_iter=3000, random_state=RANDOM_STATE,
        ),
        "XGBoost": _xgb()(
            colsample_bytree=0.8, gamma=1, learning_rate=0.05,
            max_depth=2, n_estimators=50, subsample=0.8,
            reg_lambda=1, n_jobs=-1, eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }
    return models


def get_best_model():
    """Return the tuned XGBoost (hyperparameters from GridSearchCV)."""
    return _xgb()(**BEST_MODEL_PARAMS)
