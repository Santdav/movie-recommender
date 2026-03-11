import os

import pandas as pd
import joblib

PROCESSED_DIR = "data/processed"


def time_based_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("timestamp column must be datetime. Ensure load_ratings() was used.")

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_ratio))

    train = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test = df_sorted.iloc[split_idx:].reset_index(drop=True)

    print(f"Train : {len(train):,} ratings  ({train['timestamp'].min().date()} => {train['timestamp'].max().date()})")
    print(f"Test  : {len(test):,} ratings  ({test['timestamp'].min().date()} => {test['timestamp'].max().date()})")
    _warn_cold_start(train, test)

    return train, test


def _warn_cold_start(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> None:
    
    train_users = set(train["userId"].unique())
    train_movies = set(train["movieId"].unique())

    unseen_users = set(test["userId"].unique()) - train_users
    unseen_movies = set(test["movieId"].unique()) - train_movies

    if unseen_users:
        print(f"Warning: {len(unseen_users)} users in test have no training ratings (cold start).")
    if unseen_movies:
        print(f"Warning: {len(unseen_movies)} movies in test have no training ratings (cold start).")


def save_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train.to_parquet(os.path.join(PROCESSED_DIR, "train.parquet"), index=False)
    test.to_parquet(os.path.join(PROCESSED_DIR, "test.parquet"), index=False)
    print(f"Splits saved to {PROCESSED_DIR}/")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:

    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
    return train, test


def split(
    df: pd.DataFrame,
    test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    train, test = time_based_split(df, test_ratio)
    save_splits(train, test)
    return train, test