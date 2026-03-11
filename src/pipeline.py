import os

import pandas as pd # type: ignore
from scipy.sparse import csr_matrix # type: ignore

from data_loader import load_all
from encoder import encode
from splitter import split
from matrix_builder import build


def run_pipeline(test_ratio: float = 0.2) -> dict:
    print("=" * 50)
    print("STEP 1 - Data Acquisition & Loading")
    print("=" * 50)
    ratings, movies, users = load_all()

    print("\n" + "=" * 50)
    print("STEP 2 - ID Encoding")
    print("=" * 50)
    encoded_ratings, user_encoder, movie_encoder, user_decoder, movie_decoder = encode(ratings)

    print("\n" + "=" * 50)
    print("STEP 3 - Train / Test Split")
    print("=" * 50)
    train, test = split(encoded_ratings, test_ratio=test_ratio)

    print("\n" + "=" * 50)
    print("STEP 4 - Building User-Item Matrix")
    print("=" * 50)
    n_users = len(user_encoder)
    n_movies = len(movie_encoder)
    matrix = build(train, n_users, n_movies)

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)
    _log_summary(ratings, train, test, matrix)

    return {
        "ratings": ratings,
        "movies": movies,
        "users": users,
        "train": train,
        "test": test,
        "matrix": matrix,
        "user_encoder": user_encoder,
        "movie_encoder": movie_encoder,
        "user_decoder": user_decoder,
        "movie_decoder": movie_decoder,
    }


def _log_summary(
    ratings: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    matrix: csr_matrix
) -> None:
    print(f"Total ratings   : {len(ratings):,}")
    print(f"Train ratings   : {len(train):,}  ({len(train) / len(ratings):.1%})")
    print(f"Test ratings    : {len(test):,}  ({len(test) / len(ratings):.1%})")
    print(f"Matrix shape    : {matrix.shape[0]:,} x {matrix.shape[1]:,}")
    print(f"Artifacts saved : data/processed/")


if __name__ == "__main__":
    run_pipeline()