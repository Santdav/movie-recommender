import pytest
import pandas as pd
import numpy as np


from src.encoder import build_encoders, apply_encoders


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_ratings() -> pd.DataFrame:
    return pd.DataFrame({
        "userId":    [1, 1, 2, 2, 3],
        "movieId":   [10, 20, 10, 30, 20],
        "rating":    [5.0, 3.0, 4.0, 2.0, 5.0],
        "timestamp": pd.to_datetime([
            "2000-01-01", "2000-01-02", "2000-01-03",
            "2000-01-04", "2000-01-05"
        ])
    })


@pytest.fixture(scope="module")
def encoders(sample_ratings: pd.DataFrame):
    return build_encoders(sample_ratings)


# ─── build_encoders ───────────────────────────────────────────────────────────

class TestBuildEncoders:
    def test_user_encoder_covers_all_users(
        self, sample_ratings: pd.DataFrame, encoders: tuple
    ) -> None:
        user_encoder, _, _, _ = encoders
        assert set(user_encoder.keys()) == set(sample_ratings["userId"].unique())

    def test_movie_encoder_covers_all_movies(
        self, sample_ratings: pd.DataFrame, encoders: tuple
    ) -> None:
        _, movie_encoder, _, _ = encoders
        assert set(movie_encoder.keys()) == set(sample_ratings["movieId"].unique())

    def test_user_indices_are_contiguous(self, encoders: tuple) -> None:
        user_encoder, _, _, _ = encoders
        indices = sorted(user_encoder.values())
        assert indices == list(range(len(indices)))

    def test_movie_indices_are_contiguous(self, encoders: tuple) -> None:
        _, movie_encoder, _, _ = encoders
        indices = sorted(movie_encoder.values())
        assert indices == list(range(len(indices)))

    def test_user_decoder_is_inverse_of_encoder(self, encoders: tuple) -> None:
        user_encoder, _, user_decoder, _ = encoders
        for uid, idx in user_encoder.items():
            assert user_decoder[idx] == uid

    def test_movie_decoder_is_inverse_of_encoder(self, encoders: tuple) -> None:
        _, movie_encoder, _, movie_decoder = encoders
        for mid, idx in movie_encoder.items():
            assert movie_decoder[idx] == mid

    def test_encoding_is_deterministic(self, sample_ratings: pd.DataFrame) -> None:
        enc1, _, _, _ = build_encoders(sample_ratings)
        enc2, _, _, _ = build_encoders(sample_ratings)
        assert enc1 == enc2


# ─── apply_encoders ───────────────────────────────────────────────────────────

class TestApplyEncoders:
    def test_adds_user_idx_column(
        self, sample_ratings: pd.DataFrame, encoders: tuple
    ) -> None:
        user_encoder, movie_encoder, _, _ = encoders
        result = apply_encoders(sample_ratings, user_encoder, movie_encoder)
        assert "user_idx" in result.columns

    def test_adds_movie_idx_column(
        self, sample_ratings: pd.DataFrame, encoders: tuple
    ) -> None:
        user_encoder, movie_encoder, _, _ = encoders
        result = apply_encoders(sample_ratings, user_encoder, movie_encoder)
        assert "movie_idx" in result.columns

    def test_indices_are_integers(
        self, sample_ratings: pd.DataFrame, encoders: tuple
    ) -> None:
        user_encoder, movie_encoder, _, _ = encoders
        result = apply_encoders(sample_ratings, user_encoder, movie_encoder)
        assert pd.api.types.is_integer_dtype(result["user_idx"])
        assert pd.api.types.is_integer_dtype(result["movie_idx"])

    def test_no_nulls_in_indices(
        self, sample_ratings: pd.DataFrame, encoders: tuple
    ) -> None:
        user_encoder, movie_encoder, _, _ = encoders
        result = apply_encoders(sample_ratings, user_encoder, movie_encoder)
        assert result["user_idx"].isnull().sum() == 0
        assert result["movie_idx"].isnull().sum() == 0

    def test_does_not_mutate_input(
        self, sample_ratings: pd.DataFrame, encoders: tuple
    ) -> None:
        user_encoder, movie_encoder, _, _ = encoders
        original_columns = list(sample_ratings.columns)
        apply_encoders(sample_ratings, user_encoder, movie_encoder)
        assert list(sample_ratings.columns) == original_columns

    def test_raises_on_unknown_user(self, encoders: tuple) -> None:
        user_encoder, movie_encoder, _, _ = encoders
        bad_df = pd.DataFrame({
            "userId":  [999],
            "movieId": [10],
            "rating":  [3.0]
        })
        with pytest.raises(ValueError, match="Encoding failed"):
            apply_encoders(bad_df, user_encoder, movie_encoder)

    def test_raises_on_unknown_movie(self, encoders: tuple) -> None:
        user_encoder, movie_encoder, _, _ = encoders
        bad_df = pd.DataFrame({
            "userId":  [1],
            "movieId": [999],
            "rating":  [3.0]
        })
        with pytest.raises(ValueError, match="Encoding failed"):
            apply_encoders(bad_df, user_encoder, movie_encoder)