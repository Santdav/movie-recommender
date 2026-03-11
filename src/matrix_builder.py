import os

import joblib # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from scipy.sparse import csr_matrix # pyright: ignore[reportMissingImports]

PROCESSED_DIR = "data/processed"
MATRIX_PATH = os.path.join(PROCESSED_DIR, "interaction_matrix.joblib")


def build_sparse_matrix(
    df: pd.DataFrame,
    n_users: int,
    n_movies: int
) -> csr_matrix:
    
    _validate_encoded(df)

    matrix = csr_matrix(
        (df["rating"].values.astype(np.float32), (df["user_idx"].values, df["movie_idx"].values)),
        shape=(n_users, n_movies)
    )

    _log_matrix_stats(matrix)
    return matrix


def _validate_encoded(df: pd.DataFrame) -> None:
    for col in ("user_idx", "movie_idx", "rating"):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Ensure encode() was called before building the matrix."
            )
    if df["user_idx"].isna().any() or df["movie_idx"].isna().any():
        raise ValueError("NaN values found in user_idx or movie_idx. Encoding may have failed.")


def _log_matrix_stats(matrix: csr_matrix) -> None:
    n_users, n_movies = matrix.shape
    n_ratings = matrix.nnz
    sparsity = 1 - (n_ratings / (n_users * n_movies))

    print(f"Matrix shape : {n_users:,} users x {n_movies:,} movies")
    print(f"Stored ratings : {n_ratings:,}")
    print(f"Sparsity : {sparsity:.4%}")

    ratings_per_user = np.diff(matrix.indptr)
    print(f"Ratings per user  - mean: {ratings_per_user.mean():.1f}  min: {ratings_per_user.min()}  max: {ratings_per_user.max()}")

    ratings_per_movie = np.diff(matrix.tocsc().indptr)
    print(f"Ratings per movie - mean: {ratings_per_movie.mean():.1f}  min: {ratings_per_movie.min()}  max: {ratings_per_movie.max()}")


def save_matrix(matrix: csr_matrix) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump(matrix, MATRIX_PATH)
    print(f"Matrix saved to {MATRIX_PATH}")


def load_matrix() -> csr_matrix:
    return joblib.load(MATRIX_PATH)


def build(
    train_df: pd.DataFrame,
    n_users: int,
    n_movies: int
) -> csr_matrix:
    matrix = build_sparse_matrix(train_df, n_users, n_movies)
    save_matrix(matrix)
    return matrix