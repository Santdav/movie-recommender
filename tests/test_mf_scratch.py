import pytest
import numpy as np
from scipy.sparse import csr_matrix


from src.mf_scratch import MatrixFactorizationSGD


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_matrix() -> csr_matrix:
    """
    5 users x 4 movies with known rating patterns.
    Users 0 and 1 like movies 0 and 1.
    Users 2 and 3 like movies 2 and 3.
    """
    data = np.array([5, 4, 5, 4, 5, 4, 4, 5], dtype=np.float32)
    row  = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    col  = np.array([0, 1, 0, 1, 2, 3, 2, 3])
    return csr_matrix((data, (row, col)), shape=(5, 4))


@pytest.fixture(scope="module")
def fitted_model(sample_matrix: csr_matrix) -> MatrixFactorizationSGD:
    return MatrixFactorizationSGD(k=5, n_epochs=50, random_state=42).fit(sample_matrix)


# ─── Fitting ─────────────────────────────────────────────────────────────────

class TestFit:
    def test_is_fitted_after_fit(self, fitted_model: MatrixFactorizationSGD) -> None:
        assert fitted_model.is_fitted

    def test_P_shape(
        self, fitted_model: MatrixFactorizationSGD, sample_matrix: csr_matrix
    ) -> None:
        assert fitted_model._P.shape == (sample_matrix.shape[0], fitted_model.k)

    def test_Q_shape(
        self, fitted_model: MatrixFactorizationSGD, sample_matrix: csr_matrix
    ) -> None:
        assert fitted_model._Q.shape == (sample_matrix.shape[1], fitted_model.k)

    def test_user_bias_shape(
        self, fitted_model: MatrixFactorizationSGD, sample_matrix: csr_matrix
    ) -> None:
        assert fitted_model._b_u.shape == (sample_matrix.shape[0],)

    def test_item_bias_shape(
        self, fitted_model: MatrixFactorizationSGD, sample_matrix: csr_matrix
    ) -> None:
        assert fitted_model._b_i.shape == (sample_matrix.shape[1],)

    def test_mu_is_global_mean(
        self, fitted_model: MatrixFactorizationSGD, sample_matrix: csr_matrix
    ) -> None:
        expected_mu = sample_matrix.data.mean()
        assert abs(fitted_model._mu - expected_mu) < 1e-4

    def test_train_loss_has_correct_length(
        self, fitted_model: MatrixFactorizationSGD
    ) -> None:
        assert len(fitted_model.train_loss) == fitted_model.n_epochs

    def test_train_loss_decreases(self, fitted_model: MatrixFactorizationSGD) -> None:
        losses = fitted_model.train_loss
        assert losses[0] > losses[-1]

    def test_fit_returns_self(self, sample_matrix: csr_matrix) -> None:
        model = MatrixFactorizationSGD(k=5, n_epochs=5, random_state=42)
        result = model.fit(sample_matrix)
        assert result is model

    def test_not_fitted_before_fit(self) -> None:
        model = MatrixFactorizationSGD()
        assert not model.is_fitted


# ─── Predict ─────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_float(self, fitted_model: MatrixFactorizationSGD) -> None:
        result = fitted_model.predict_rating(0, 0)
        assert isinstance(result, float)

    def test_prediction_in_valid_range(
        self, fitted_model: MatrixFactorizationSGD
    ) -> None:
        result = fitted_model.predict_rating(0, 0)
        assert 1.0 <= result <= 5.0

    def test_known_high_rating_predicted_high(
        self, fitted_model: MatrixFactorizationSGD
    ) -> None:
        # user 0 rated movie 0 with 5 — prediction should be close to high
        result = fitted_model.predict_rating(0, 0)
        assert result >= 3.0

    def test_raises_when_not_fitted(self) -> None:
        model = MatrixFactorizationSGD()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_rating(0, 0)

    def test_prediction_clipped_to_max(
        self, sample_matrix: csr_matrix
    ) -> None:
        model = MatrixFactorizationSGD(k=5, n_epochs=1, random_state=42)
        model.fit(sample_matrix)
        result = model.predict_rating(0, 0)
        assert result <= 5.0

    def test_prediction_clipped_to_min(
        self, sample_matrix: csr_matrix
    ) -> None:
        model = MatrixFactorizationSGD(k=5, n_epochs=1, random_state=42)
        model.fit(sample_matrix)
        result = model.predict_rating(0, 0)
        assert result >= 1.0


# ─── Recommend ───────────────────────────────────────────────────────────────

class TestRecommend:
    def test_returns_list(self, fitted_model: MatrixFactorizationSGD) -> None:
        result = fitted_model.recommend(0, n=3)
        assert isinstance(result, list)

    def test_returns_tuples_of_int_and_float(
        self, fitted_model: MatrixFactorizationSGD
    ) -> None:
        result = fitted_model.recommend(0, n=3)
        for movie_idx, score in result:
            assert isinstance(movie_idx, int)
            assert isinstance(score, float)

    def test_respects_n_limit(self, fitted_model: MatrixFactorizationSGD) -> None:
        result = fitted_model.recommend(0, n=2)
        assert len(result) <= 2

    def test_results_sorted_descending(
        self, fitted_model: MatrixFactorizationSGD
    ) -> None:
        result = fitted_model.recommend(0, n=10)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_rated_mask_excludes_items(
        self, fitted_model: MatrixFactorizationSGD, sample_matrix: csr_matrix
    ) -> None:
        user_idx = 0
        rated = sample_matrix[user_idx].nonzero()[1]
        mask = np.zeros(sample_matrix.shape[1], dtype=bool)
        mask[rated] = True

        result = fitted_model.recommend(user_idx, n=10, rated_mask=mask)
        recommended_ids = {mid for mid, _ in result}
        assert recommended_ids.isdisjoint(set(rated.tolist()))

    def test_raises_when_not_fitted(self) -> None:
        model = MatrixFactorizationSGD()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.recommend(0, n=5)