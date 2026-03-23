import os
import zipfile
import urllib.request

import pandas as pd

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

RAW_DIR = "data/raw"
EXTRACTED_DIR = os.path.join(RAW_DIR, "ml-1m")
EXTRACTED_DIR_25M = os.path.join(RAW_DIR, "ml-25m")

ZIP_PATH = os.path.join(RAW_DIR, "ml-1m.zip")
ZIP_PATH_25M = os.path.join(RAW_DIR, "ml-25m.zip")

EXPECTED_FILES = ["ratings.dat", "movies.dat", "users.dat"]
EXPECTED_FILES_25M = ["genome-scores.csv", "genome-tags.csv"]


# ─── 1M Acquisition ──────────────────────────────────────────────────────────

def _files_exist() -> bool:
    return all(
        os.path.exists(os.path.join(EXTRACTED_DIR, f)) for f in EXPECTED_FILES
    )


def acquire_data() -> None:
    if _files_exist():
        print("Raw data already present, skipping download.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    try:
        if not os.path.exists(ZIP_PATH):
            print("Downloading MovieLens 1M...")
            urllib.request.urlretrieve(MOVIELENS_1M_URL, ZIP_PATH)
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


# ─── 25M Genome Acquisition ──────────────────────────────────────────────────

def _genome_files_exist() -> bool:
    return all(
        os.path.exists(os.path.join(EXTRACTED_DIR_25M, f)) for f in EXPECTED_FILES_25M
    )


def acquire_genome() -> None:
    if _genome_files_exist():
        print("Genome files already present, skipping download.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    try:
        if not os.path.exists(ZIP_PATH_25M):
            print("Downloading MovieLens 25M (~250MB, this may take a while)...")
            urllib.request.urlretrieve(MOVIELENS_25M_URL, ZIP_PATH_25M)
            print("Download complete.")

        print("Extracting genome files...")
        with zipfile.ZipFile(ZIP_PATH_25M, "r") as z:
            genome_files = [
                f for f in z.namelist()
                if any(f.endswith(gf) for gf in EXPECTED_FILES_25M)
            ]
            for f in genome_files:
                z.extract(f, RAW_DIR)
        print(f"Genome files extracted to {EXTRACTED_DIR_25M}")

    except Exception as e:
        print(f"Automated download failed: {e}")
        print(
            "Please download manually from: https://grouplens.org/datasets/movielens/25m/"
            f"\nPlace genome-scores.csv and genome-tags.csv inside {EXTRACTED_DIR_25M}/"
        )
        raise


# ─── 1M Loaders ──────────────────────────────────────────────────────────────

def load_ratings() -> pd.DataFrame:
    path = os.path.join(EXTRACTED_DIR, "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def load_movies() -> pd.DataFrame:
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


def load_users() -> pd.DataFrame:
    path = os.path.join(EXTRACTED_DIR, "users.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["userId", "gender", "age", "occupation", "zipcode"]
    )
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acquire_data()
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()
    return ratings, movies, users


# ─── Genome Loaders ──────────────────────────────────────────────────────────

def load_genome_scores() -> pd.DataFrame:
    """
    Loads the tag genome relevance scores.
    Returns a DataFrame with columns: movieId, tagId, relevance.
    Relevance is a float in [0, 1] indicating how relevant a tag is to a movie.
    """
    path = os.path.join(EXTRACTED_DIR_25M, "genome-scores.csv")
    return pd.read_csv(path)


def load_genome_tags() -> pd.DataFrame:
    """
    Loads the tag genome tag definitions.
    Returns a DataFrame with columns: tagId, tag.
    """
    path = os.path.join(EXTRACTED_DIR_25M, "genome-tags.csv")
    return pd.read_csv(path)


def load_genome() -> tuple[pd.DataFrame, pd.DataFrame]:
    acquire_genome()
    scores = load_genome_scores()
    tags = load_genome_tags()
    print(f"Genome scores : {scores.shape}")
    print(f"Genome tags   : {tags.shape}")
    return scores, tags


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ratings, movies, users = load_all()
    print(f"Ratings : {ratings.shape}")
    print(f"Movies  : {movies.shape}")
    print(f"Users   : {users.shape}")
    print("\nRatings sample:")
    print(ratings.head())
    print("\nMovies sample:")
    print(movies.head())