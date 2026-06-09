"""Feature preprocessing pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

from app.config import (
    NUMERIC_FEATURES,
    NUMERIC_FEATURES_AGE,
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
)


def create_preprocessor():
    """Build the ColumnTransformer with:
    - PowerTransform (Yeo-Johnson) + StandardScaler for right-skewed numerics
    - StandardScaler only for age
    - OneHotEncoder for nominal categoricals
    - OrdinalEncoder for ordinal categoricals
    """
    num_pipe = Pipeline(steps=[
        ("yeo_johnson", PowerTransformer(method="yeo-johnson")),
        ("scaler", StandardScaler()),
    ])

    num_pipe_age = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("ohe", OneHotEncoder(
            sparse_output=False, drop="first", handle_unknown="ignore",
        )),
    ])

    ord_pipe = Pipeline(steps=[
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numpipe", num_pipe, NUMERIC_FEATURES),
            ("numpipe_age", num_pipe_age, NUMERIC_FEATURES_AGE),
            ("catpipe", cat_pipe, CATEGORICAL_FEATURES),
            ("ordpipe", ord_pipe, ORDINAL_FEATURES),
        ],
        remainder="passthrough",
    )
    return preprocessor


def get_feature_names(fitted_preprocessor):
    """Return feature names after the preprocessor has been fitted."""
    try:
        return list(fitted_preprocessor.get_feature_names_out())
    except Exception:
        return None
