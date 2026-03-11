import os
import zipfile
import urllib.request

import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_DIR = "data/raw"
EXTRACTED_DIR = os.path.join(RAW_DIR, "ml-1m")
ZIP_PATH = os.path.join(RAW_DIR, "ml-1m.zip")

EXPECTED_FILES = ["ratings.dat", "movies.dat", "users.dat"]


def _files_exist():
    return all(
        os.path.exists(os.path.join(EXTRACTED_DIR, f)) for f in EXPECTED_FILES
    )


def acquire_data():
    if _files_exist():
        print("Raw data already present, skipping download.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    try:
        if not os.path.exists(ZIP_PATH):
            print("Downloading MovieLens 1M...")
            urllib.request.urlretrieve(MOVIELENS_URL, ZIP_PATH)
            print("Download complete.")

        print("Extracting files...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(RAW_DIR)
        print(f"Files extracted to {EXTRACTED_DIR}")

    except Exception as e:
        print(f"Automated download failed: {e}")
        print(
            "Please download manually from: https://grouplens.org/datasets/movielens/1m/"
            f"\nPlace the extracted ml-1m/ folder inside {RAW_DIR}/"
        )
        raise


def load_ratings():
    path = os.path.join(EXTRACTED_DIR, "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def load_movies():
    path = os.path.join(EXTRACTED_DIR, "movies.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1"
    )
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("Int64")
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
    df["genres"] = df["genres"].str.split("|")
    return df


def load_users():
    path = os.path.join(EXTRACTED_DIR, "users.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["userId", "gender", "age", "occupation", "zipcode"]
    )
    return df


def load_all():
    acquire_data()
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()
    return ratings, movies, users


if __name__ == "__main__":
    acquire_data()
    ratings, movies, users = load_all()
    print(f"Ratings : {ratings.shape}")
    print(f"Movies  : {movies.shape}")
    print(f"Users   : {users.shape}")
    print("\nRatings sample:")
    print(ratings.head())
    print("\nMovies sample:")
    print(movies.head())