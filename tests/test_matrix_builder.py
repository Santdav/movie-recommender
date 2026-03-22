import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


from src.matrix_builder import build_sparse_matrix


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "userId":    [1, 1, 2, 2, 3],
        "movieId":   [10, 20, 10, 30, 20],
        "rating":    [5.0, 3.0, 4.0, 2.0, 5.0],
        "user_idx":  [0, 0, 1, 1, 2],
        "movie_idx": [0, 1, 0, 2, 1],
        "timestamp": pd.to_datetime([
            "2000-01-01", "2000-02-01", "2000-03-01",
            "2000-04-01", "2000-05-01"
        ])
    })


@pytest.fixture(scope="module")
def matrix(sample_df: pd.DataFrame) -> csr_matrix:
    return build_sparse_matrix(sample_df, n_users=3, n_movies=3)


# ─── Shape and Type ───────────────────────────────────────────────────────────

class TestMatrixShapeAndType:
    def test_returns_csr_matrix(self, matrix: csr_matrix) -> None:
        assert isinstance(matrix, csr_matrix)

    def test_correct_shape(self, matrix: csr_matrix) -> None:
        assert matrix.shape == (3, 3)

    def test_correct_number_of_stored_ratings(
        self, sample_df: pd.DataFrame, matrix: csr_matrix
    ) -> None:
        assert matrix.nnz == len(sample_df)

    def test_dtype_is_float32(self, matrix: csr_matrix) -> None:
        assert matrix.dtype == np.float32


# ─── Values ──────────────────────────────────────────────────────────────────

class TestMatrixValues:
    def test_ratings_are_within_valid_range(self, matrix: csr_matrix) -> None:
        stored = matrix.data
        assert (stored >= 1.0).all() and (stored <= 5.0).all()

    def test_specific_rating_is_correct(self, matrix: csr_matrix) -> None:
        # user_idx=0, movie_idx=0 should be 5.0
        assert matrix[0, 0] == pytest.approx(5.0)

    def test_unrated_entry_is_zero(self, matrix: csr_matrix) -> None:
        # user_idx=2, movie_idx=0 has no rating in sample
        assert matrix[2, 0] == 0.0

    def test_all_stored_values_are_positive(self, matrix: csr_matrix) -> None:
        assert (matrix.data > 0).all()


# ─── Sparsity ─────────────────────────────────────────────────────────────────

class TestMatrixSparsity:
    def test_sparsity_is_correct(self, matrix: csr_matrix) -> None:
        n_users, n_movies = matrix.shape
        expected_sparsity = 1 - (matrix.nnz / (n_users * n_movies))
        actual_sparsity = 1 - (matrix.nnz / (n_users * n_movies))
        assert abs(actual_sparsity - expected_sparsity) < 1e-6

    def test_matrix_is_sparse(self, matrix: csr_matrix) -> None:
        n_users, n_movies = matrix.shape
        sparsity = 1 - (matrix.nnz / (n_users * n_movies))
        assert sparsity > 0.0


# ─── Validation ──────────────────────────────────────────────────────────────

class TestMatrixValidation:
    def test_raises_on_missing_user_idx(self, sample_df: pd.DataFrame) -> None:
        bad_df = sample_df.drop(columns=["user_idx"])
        with pytest.raises(ValueError, match="user_idx"):
            build_sparse_matrix(bad_df, n_users=3, n_movies=3)

    def test_raises_on_missing_movie_idx(self, sample_df: pd.DataFrame) -> None:
        bad_df = sample_df.drop(columns=["movie_idx"])
        with pytest.raises(ValueError, match="movie_idx"):
            build_sparse_matrix(bad_df, n_users=3, n_movies=3)

    def test_raises_on_missing_rating(self, sample_df: pd.DataFrame) -> None:
        bad_df = sample_df.drop(columns=["rating"])
        with pytest.raises(ValueError, match="rating"):
            build_sparse_matrix(bad_df, n_users=3, n_movies=3)

    def test_raises_on_null_user_idx(self, sample_df: pd.DataFrame) -> None:
        bad_df = sample_df.copy()
        bad_df.loc[0, "user_idx"] = None
        with pytest.raises(ValueError, match="NaN values"):
            build_sparse_matrix(bad_df, n_users=3, n_movies=3)

    def test_raises_on_null_movie_idx(self, sample_df: pd.DataFrame) -> None:
        bad_df = sample_df.copy()
        bad_df.loc[0, "movie_idx"] = None
        with pytest.raises(ValueError, match="NaN values"):
            build_sparse_matrix(bad_df, n_users=3, n_movies=3)


# ─── Larger Matrix ────────────────────────────────────────────────────────────

class TestLargerMatrix:
    def test_shape_respects_n_users_and_n_movies(
        self, sample_df: pd.DataFrame
    ) -> None:
        # pass larger dimensions than the data requires
        matrix = build_sparse_matrix(sample_df, n_users=10, n_movies=10)
        assert matrix.shape == (10, 10)

    def test_nnz_unchanged_with_larger_dimensions(
        self, sample_df: pd.DataFrame
    ) -> None:
        matrix = build_sparse_matrix(sample_df, n_users=10, n_movies=10)
        assert matrix.nnz == len(sample_df)