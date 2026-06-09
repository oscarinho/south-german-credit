"""Training utilities: CV, model comparison, best-model training, persistence."""
import joblib
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler

from app.config import (
    RANDOM_STATE,
    CV_SPLITS,
    CV_REPEATS,
    SCORING_METRICS,
    MODEL_PATH,
    MODELS_DIR,
)
from app.preprocessing import create_preprocessor
from app.models import get_all_models, get_best_model


RESAMPLERS = {
    "None": None,
    "RandomUnder": RandomUnderSampler(random_state=RANDOM_STATE),
    "SMOTE": SMOTE(random_state=RANDOM_STATE, k_neighbors=5),
    # KMeansSMOTE silently produces NaN when minority clusters are too small
    # for the default cluster_balance_threshold. We relax it and reduce
    # k_neighbors to make it work on n~850.
    "KMeansSMOTE": KMeansSMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=3,
        cluster_balance_threshold=0.1,
    ),
    "SMOTETomek": SMOTETomek(random_state=RANDOM_STATE),
}


def create_training_pipeline(model, resampler=None):
    """Build a (preprocessor -> [resampler] -> model) pipeline."""
    pre = create_preprocessor()
    if resampler is None:
        return SkPipeline(steps=[("ct", pre), ("clf", model)])
    return ImbPipeline(steps=[
        ("ct", pre),
        ("resampler", resampler),
        ("clf", model),
    ])


def _cv():
    return RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )


def cross_validate_model(pipeline, X, y, scoring=None):
    scoring = scoring or SCORING_METRICS
    return cross_validate(
        pipeline, X, y,
        scoring=scoring, cv=_cv(),
        n_jobs=-1, return_train_score=True,
    )


def compare_models(X, y, include_resampler=False, resampler_name="SMOTETomek"):
    """Cross-validate every model in get_all_models().

    If include_resampler=True, applies the named resampler inside an
    imbalanced-learn pipeline so resampling only happens on the training fold.
    """
    resampler = RESAMPLERS.get(resampler_name) if include_resampler else None
    models = get_all_models()
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        pipe = create_training_pipeline(model, resampler=resampler)
        results[name] = cross_validate_model(pipe, X, y)
    return results


def compare_resamplers(X, y, resamplers=None, model=None):
    """Cross-validate the same model under different resamplers."""
    from app.models import get_best_model as _get_best
    model = model or _get_best()
    resamplers = resamplers or ["None", "RandomUnder", "SMOTE",
                                "KMeansSMOTE", "SMOTETomek"]
    results = {}
    for name in resamplers:
        print(f"Evaluating resampler: {name}...")
        r = RESAMPLERS.get(name)
        pipe = create_training_pipeline(model, resampler=r)
        results[name] = cross_validate_model(pipe, X, y)
    return results


def apply_smote_tomek(X, y):
    """Apply SMOTETomek and return resampled data."""
    smt = SMOTETomek(random_state=RANDOM_STATE)
    return smt.fit_resample(X, y)


def train_best_model(X, y, apply_resampling=True):
    """Train the final pipeline (preprocessor + SMOTETomek + XGBoost)."""
    resampler = SMOTETomek(random_state=RANDOM_STATE) if apply_resampling else None
    pipe = create_training_pipeline(get_best_model(), resampler=resampler)
    pipe.fit(X, y)
    return pipe


def save_model(pipeline, path=None):
    path = path or MODEL_PATH
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path=None):
    path = path or MODEL_PATH
    return joblib.load(path)
