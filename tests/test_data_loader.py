import pytest 
import pandas as pd


from src.data_loader import load_ratings, load_movies, load_users, load_all


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ratings() -> pd.DataFrame:
    return load_ratings()


@pytest.fixture(scope="module")
def movies() -> pd.DataFrame:
    return load_movies()


@pytest.fixture(scope="module")
def users() -> pd.DataFrame:
    return load_users()


# ─── Ratings ─────────────────────────────────────────────────────────────────

class TestLoadRatings:
    def test_has_expected_columns(self, ratings: pd.DataFrame) -> None:
        assert list(ratings.columns) == ["userId", "movieId", "rating", "timestamp"]

    def test_timestamp_is_datetime(self, ratings: pd.DataFrame) -> None:
        assert pd.api.types.is_datetime64_any_dtype(ratings["timestamp"])

    def test_ratings_are_in_valid_range(self, ratings: pd.DataFrame) -> None:
        assert ratings["rating"].between(1, 5).all()

    def test_no_null_values(self, ratings: pd.DataFrame) -> None:
        assert ratings.isnull().sum().sum() == 0

    def test_expected_row_count(self, ratings: pd.DataFrame) -> None:
        assert len(ratings) == 1_000_209

    def test_user_ids_are_positive(self, ratings: pd.DataFrame) -> None:
        assert (ratings["userId"] > 0).all()

    def test_movie_ids_are_positive(self, ratings: pd.DataFrame) -> None:
        assert (ratings["movieId"] > 0).all()


# ─── Movies ──────────────────────────────────────────────────────────────────

class TestLoadMovies:
    def test_has_expected_columns(self, movies: pd.DataFrame) -> None:
        for col in ["movieId", "title", "genres", "year", "title_clean"]:
            assert col in movies.columns

    def test_genres_is_list(self, movies: pd.DataFrame) -> None:
        assert movies["genres"].apply(lambda x: isinstance(x, list)).all()

    def test_genres_are_not_empty(self, movies: pd.DataFrame) -> None:
        assert movies["genres"].apply(lambda x: len(x) > 0).all()

    def test_year_is_numeric(self, movies: pd.DataFrame) -> None:
        valid_years = movies["year"].dropna()
        assert pd.api.types.is_integer_dtype(valid_years)

    def test_year_in_reasonable_range(self, movies: pd.DataFrame) -> None:
        valid_years = movies["year"].dropna()
        assert valid_years.between(1900, 2030).all()

    def test_title_clean_has_no_year_suffix(self, movies: pd.DataFrame) -> None:
        import re
        has_year = movies["title_clean"].str.contains(r"\(\d{4}\)$", regex=True)
        assert not has_year.any()

    def test_no_null_movie_ids(self, movies: pd.DataFrame) -> None:
        assert movies["movieId"].isnull().sum() == 0

    def test_expected_row_count(self, movies: pd.DataFrame) -> None:
        assert len(movies) == 3_883


# ─── Users ───────────────────────────────────────────────────────────────────

class TestLoadUsers:
    def test_has_expected_columns(self, users: pd.DataFrame) -> None:
        assert list(users.columns) == ["userId", "gender", "age", "occupation", "zipcode"]

    def test_gender_values_are_valid(self, users: pd.DataFrame) -> None:
        assert users["gender"].isin(["M", "F"]).all()

    def test_no_null_values(self, users: pd.DataFrame) -> None:
        assert users.isnull().sum().sum() == 0

    def test_expected_row_count(self, users: pd.DataFrame) -> None:
        assert len(users) == 6_040


# ─── load_all ────────────────────────────────────────────────────────────────

class TestLoadAll:
    def test_returns_three_dataframes(self) -> None:
        result = load_all()
        assert len(result) == 3
        assert all(isinstance(df, pd.DataFrame) for df in result)

    def test_ratings_and_movies_share_movie_ids(self) -> None:
        ratings, movies, _ = load_all()
        rating_movie_ids = set(ratings["movieId"].unique())
        movie_ids = set(movies["movieId"].unique())
        assert rating_movie_ids.issubset(movie_ids)