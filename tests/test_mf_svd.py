import pytest
import numpy as np
from scipy.sparse import csr_matrix


from src.mf_svd import MatrixFactorizationSVD


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_matrix() -> csr_matrix:
    """
    5 users x 4 movies with clear cluster structure.
    Users 0 and 1 like movies 0 and 1.
    Users 2 and 3 like movies 2 and 3.
    User 4 has no ratings (cold start case).
    """
    data = np.array([5, 4, 5, 4, 5, 4, 4, 5], dtype=np.float32)
    row  = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    col  = np.array([0, 1, 0, 1, 2, 3, 2, 3])
    return csr_matrix((data, (row, col)), shape=(5, 4))


@pytest.fixture(scope="module")
def fitted_model(sample_matrix: csr_matrix) -> MatrixFactorizationSVD:
    return MatrixFactorizationSVD(k=2, random_state=42).fit(sample_matrix)


# ─── Fitting ─────────────────────────────────────────────────────────────────

class TestFit:
    def test_is_fitted_after_fit(self, fitted_model: MatrixFactorizationSVD) -> None:
        assert fitted_model.is_fitted

    def test_predictions_matrix_shape(
        self, fitted_model: MatrixFactorizationSVD, sample_matrix: csr_matrix
    ) -> None:
        assert fitted_model._predictions.shape == sample_matrix.shape

    def test_U_shape(
        self, fitted_model: MatrixFactorizationSVD, sample_matrix: csr_matrix
    ) -> None:
        assert fitted_model._U.shape[0] == sample_matrix.shape[0]

    def test_Vt_shape(
        self, fitted_model: MatrixFactorizationSVD, sample_matrix: csr_matrix
    ) -> None:
        assert fitted_model._Vt.shape[1] == sample_matrix.shape[1]

    def test_sigma_has_correct_length(self, fitted_model: MatrixFactorizationSVD) -> None:
        assert len(fitted_model._sigma) <= fitted_model.k

    def test_singular_values_are_positive(
        self, fitted_model: MatrixFactorizationSVD
    ) -> None:
        assert (fitted_model._sigma > 0).all()

    def test_user_means_shape(
        self, fitted_model: MatrixFactorizationSVD, sample_matrix: csr_matrix
    ) -> None:
        assert fitted_model._user_means.shape == (sample_matrix.shape[0],)

    def test_user_means_are_non_negative(
        self, fitted_model: MatrixFactorizationSVD
    ) -> None:
        assert (fitted_model._user_means >= 0).all()

    def test_predictions_clipped_to_valid_range(
        self, fitted_model: MatrixFactorizationSVD
    ) -> None:
        preds = fitted_model._predictions
        assert (preds >= 1.0).all() and (preds <= 5.0).all()

    def test_fit_returns_self(self, sample_matrix: csr_matrix) -> None:
        model = MatrixFactorizationSVD(k=2, random_state=42)
        result = model.fit(sample_matrix)
        assert result is model

    def test_not_fitted_before_fit(self) -> None:
        model = MatrixFactorizationSVD()
        assert not model.is_fitted

    def test_k_capped_below_matrix_min_dim(self) -> None:
        # k must be strictly less than min(n_users, n_items)
        # passing k larger than allowed should not raise — model should cap it
        small_matrix = csr_matrix(
            np.array([[5, 4], [3, 2], [1, 5]], dtype=np.float32)
        )
        model = MatrixFactorizationSVD(k=100, random_state=42)
        model.fit(small_matrix)
        assert model.is_fitted


# ─── Predict ─────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_float(self, fitted_model: MatrixFactorizationSVD) -> None:
        result = fitted_model.predict_rating(0, 0)
        assert isinstance(result, float)

    def test_prediction_in_valid_range(
        self, fitted_model: MatrixFactorizationSVD
    ) -> None:
        result = fitted_model.predict_rating(0, 0)
        assert 1.0 <= result <= 5.0

    def test_known_high_rating_predicted_high(
        self, fitted_model: MatrixFactorizationSVD
    ) -> None:
        # user 0 rated movie 0 with 5 — reconstruction should be close
        result = fitted_model.predict_rating(0, 0)
        assert result >= 3.0

    def test_cluster_separation(self, fitted_model: MatrixFactorizationSVD) -> None:
        # user 0 should score movies 0/1 higher than movies 2/3
        score_in_cluster = fitted_model.predict_rating(0, 0)
        score_out_cluster = fitted_model.predict_rating(0, 2)
        assert score_in_cluster > score_out_cluster

    def test_raises_when_not_fitted(self) -> None:
        model = MatrixFactorizationSVD()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_rating(0, 0)

    def test_all_predictions_in_valid_range(
        self, fitted_model: MatrixFactorizationSVD, sample_matrix: csr_matrix
    ) -> None:
        n_users, n_items = sample_matrix.shape
        for u in range(n_users):
            for i in range(n_items):
                result = fitted_model.predict_rating(u, i)
                assert 1.0 <= result <= 5.0


# ─── Recommend ───────────────────────────────────────────────────────────────

class TestRecommend:
    def test_returns_list(self, fitted_model: MatrixFactorizationSVD) -> None:
        result = fitted_model.recommend(0, n=3)
        assert isinstance(result, list)

    def test_returns_tuples_of_int_and_float(
        self, fitted_model: MatrixFactorizationSVD
    ) -> None:
        result = fitted_model.recommend(0, n=3)
        for movie_idx, score in result:
            assert isinstance(movie_idx, int)
            assert isinstance(score, float)

    def test_respects_n_limit(self, fitted_model: MatrixFactorizationSVD) -> None:
        result = fitted_model.recommend(0, n=2)
        assert len(result) <= 2

    def test_results_sorted_descending(
        self, fitted_model: MatrixFactorizationSVD
    ) -> None:
        result = fitted_model.recommend(0, n=10)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_rated_mask_excludes_items(
        self, fitted_model: MatrixFactorizationSVD, sample_matrix: csr_matrix
    ) -> None:
        user_idx = 0
        rated = sample_matrix[user_idx].nonzero()[1]
        mask = np.zeros(sample_matrix.shape[1], dtype=bool)
        mask[rated] = True

        result = fitted_model.recommend(user_idx, n=10, rated_mask=mask)
        recommended_ids = {mid for mid, _ in result}
        assert recommended_ids.isdisjoint(set(rated.tolist()))

    def test_raises_when_not_fitted(self) -> None:
        model = MatrixFactorizationSVD()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.recommend(0, n=5)

    def test_recommend_without_mask_returns_all_items(
        self, fitted_model: MatrixFactorizationSVD, sample_matrix: csr_matrix
    ) -> None:
        # without a mask, all items including rated ones are returned
        result = fitted_model.recommend(0, n=sample_matrix.shape[1])
        assert len(result) == sample_matrix.shape[1]