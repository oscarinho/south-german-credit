"""Data loading and splitting utilities."""
import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import (
    DATA_PATH,
    COLUMN_TRANSLATION,
    TARGET,
    RANDOM_STATE,
)


def load_and_prepare_data(path=None):
    """Load the South German Credit CSV and translate columns to English.

    Note on labels: original dataset uses 1=good, 2=bad. We re-encode to
    0=good (majority) and 1=bad (minority, positive class of interest).
    """
    path = path or DATA_PATH
    # CSV is space-separated with original German column names
    df = pd.read_csv(path, sep=" ")
    df = df.rename(columns=COLUMN_TRANSLATION)

    # Re-encode target so the minority "bad credit" class is the positive (1)
    # The README says 1=good already, but the raw .asc / .csv uses 1=good/2=bad
    # We harmonize to {good:0, bad:1}.
    if set(df[TARGET].unique()) == {1, 2}:
        df[TARGET] = df[TARGET].map({1: 0, 2: 1})
    elif set(df[TARGET].unique()) == {0, 1}:
        # already encoded - check majority class is 0
        if (df[TARGET] == 0).sum() < (df[TARGET] == 1).sum():
            df[TARGET] = 1 - df[TARGET]
    return df


def split_features_target(df):
    """Separate features from target."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def create_data_splits(X, y, random_state=RANDOM_STATE):
    """70-15-15 stratified split into train / val / test."""
    X_train, X_tv, y_train, y_tv = train_test_split(
        X, y, train_size=0.70, shuffle=True,
        random_state=random_state, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tv, y_tv, train_size=0.50, shuffle=True,
        random_state=random_state, stratify=y_tv,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_train_val_combined(X_train, X_val, y_train, y_val):
    """Combine train+val for cross-validation (test stays held out)."""
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)
    return X_tv, y_tv
