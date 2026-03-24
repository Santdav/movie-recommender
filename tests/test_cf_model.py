import pytest
import numpy as np
from scipy.sparse import csr_matrix


from models.cf_model import UserBasedCF, ItemBasedCF


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_matrix() -> csr_matrix:
    """
    5 users x 4 movies. Designed so users 0 and 1 are very similar
    and users 0 and 4 are very dissimilar.

         M0   M1   M2   M3
    U0 [  5    4    0    1  ]
    U1 [  5    4    0    2  ]
    U2 [  0    0    5    4  ]
    U3 [  0    0    4    5  ]
    U4 [  1    2    0    0  ]
    """
    data = np.array([
        5, 4, 1,   # user 0
        5, 4, 2,   # user 1
        5, 4,      # user 2
        4, 5,      # user 3
        1, 2,      # user 4
    ], dtype=np.float32)

    row = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4])
    col = np.array([0, 1, 3, 0, 1, 3, 2, 3, 2, 3, 0, 1])

    return csr_matrix((data, (row, col)), shape=(5, 4))


@pytest.fixture(scope="module")
def fitted_user_cf(sample_matrix: csr_matrix) -> UserBasedCF:
    return UserBasedCF(k=2).fit(sample_matrix)


@pytest.fixture(scope="module")
def fitted_item_cf(sample_matrix: csr_matrix) -> ItemBasedCF:
    return ItemBasedCF(k=2).fit(sample_matrix)


# ─── UserBasedCF — Fitting ────────────────────────────────────────────────────

class TestUserBasedCFFit:
    def test_is_fitted_after_fit(self, fitted_user_cf: UserBasedCF) -> None:
        assert fitted_user_cf.is_fitted

    def test_similarity_matrix_shape(
        self, fitted_user_cf: UserBasedCF, sample_matrix: csr_matrix
    ) -> None:
        n = sample_matrix.shape[0]
        assert fitted_user_cf._similarity.shape == (n, n)

    def test_similarity_diagonal_is_one(self, fitted_user_cf: UserBasedCF) -> None:
        diag = np.diag(fitted_user_cf._similarity)
        np.testing.assert_allclose(diag, 1.0, atol=1e-5)

    def test_similarity_is_symmetric(self, fitted_user_cf: UserBasedCF) -> None:
        sim = fitted_user_cf._similarity
        np.testing.assert_allclose(sim, sim.T, atol=1e-5)

    def test_similarity_values_in_valid_range(self, fitted_user_cf: UserBasedCF) -> None:
        sim = fitted_user_cf._similarity
        assert (sim >= -1.0 - 1e-5).all() and (sim <= 1.0 + 1e-5).all()

    def test_similar_users_have_high_similarity(self, fitted_user_cf: UserBasedCF) -> None:
        # users 0 and 1 have nearly identical ratings
        assert fitted_user_cf._similarity[0, 1] > 0.9

    def test_dissimilar_users_have_low_similarity(self, fitted_user_cf: UserBasedCF) -> None:
        # users 0 and 2 have completely different rating patterns
        assert fitted_user_cf._similarity[0, 2] < 0.2

    def test_fit_returns_self(self, sample_matrix: csr_matrix) -> None:
        model = UserBasedCF(k=2)
        result = model.fit(sample_matrix)
        assert result is model


# ─── UserBasedCF — Predict ───────────────────────────────────────────────────

class TestUserBasedCFPredict:
    def test_returns_float(self, fitted_user_cf: UserBasedCF) -> None:
        result = fitted_user_cf.predict_rating(0, 2)
        assert isinstance(result, float)

    def test_prediction_in_valid_range(self, fitted_user_cf: UserBasedCF) -> None:
        result = fitted_user_cf.predict_rating(0, 2)
        assert 0.0 <= result <= 5.0

    def test_returns_zero_for_no_neighbor_data(
        self, fitted_user_cf: UserBasedCF
    ) -> None:
        # user 0 has k=2 neighbors: users 1 and 4
        # if neither neighbor rated movie 2, prediction should be 0.0
        result = fitted_user_cf.predict_rating(0, 2)
        assert isinstance(result, float)

    def test_raises_when_not_fitted(self, sample_matrix: csr_matrix) -> None:
        model = UserBasedCF(k=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_rating(0, 0)


# ─── UserBasedCF — Recommend ─────────────────────────────────────────────────

class TestUserBasedCFRecommend:
    def test_returns_list(self, fitted_user_cf: UserBasedCF) -> None:
        result = fitted_user_cf.recommend(0, n=5)
        assert isinstance(result, list)

    def test_returns_tuples_of_int_and_float(self, fitted_user_cf: UserBasedCF) -> None:
        result = fitted_user_cf.recommend(0, n=5)
        for movie_idx, score in result:
            assert isinstance(movie_idx, int)
            assert isinstance(score, float)

    def test_does_not_recommend_already_rated_movies(
        self, fitted_user_cf: UserBasedCF, sample_matrix: csr_matrix
    ) -> None:
        user_idx = 0
        already_rated = set(sample_matrix[user_idx].nonzero()[1])
        recommendations = fitted_user_cf.recommend(user_idx, n=10)
        recommended_ids = {mid for mid, _ in recommendations}
        assert recommended_ids.isdisjoint(already_rated)

    def test_results_are_sorted_descending(self, fitted_user_cf: UserBasedCF) -> None:
        result = fitted_user_cf.recommend(0, n=10)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_respects_n_limit(self, fitted_user_cf: UserBasedCF) -> None:
        result = fitted_user_cf.recommend(0, n=2)
        assert len(result) <= 2

    def test_raises_when_not_fitted(self) -> None:
        model = UserBasedCF(k=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.recommend(0, n=5)


# ─── ItemBasedCF — Fitting ────────────────────────────────────────────────────

class TestItemBasedCFFit:
    def test_is_fitted_after_fit(self, fitted_item_cf: ItemBasedCF) -> None:
        assert fitted_item_cf.is_fitted

    def test_similarity_matrix_shape(
        self, fitted_item_cf: ItemBasedCF, sample_matrix: csr_matrix
    ) -> None:
        n = sample_matrix.shape[1]
        assert fitted_item_cf._similarity.shape == (n, n)

    def test_similarity_is_symmetric(self, fitted_item_cf: ItemBasedCF) -> None:
        sim = fitted_item_cf._similarity
        np.testing.assert_allclose(sim, sim.T, atol=1e-5)

    def test_similar_items_have_high_similarity(self, fitted_item_cf: ItemBasedCF) -> None:
        # movies 0 and 1 are rated similarly by the same users
        assert fitted_item_cf._similarity[0, 1] > 0.5

    def test_dissimilar_items_have_low_similarity(self, fitted_item_cf: ItemBasedCF) -> None:
        # movies 0 and 2 are rated by completely different users
        assert fitted_item_cf._similarity[0, 2] < 0.2

    def test_fit_returns_self(self, sample_matrix: csr_matrix) -> None:
        model = ItemBasedCF(k=2)
        result = model.fit(sample_matrix)
        assert result is model


# ─── ItemBasedCF — Recommend ─────────────────────────────────────────────────

class TestItemBasedCFRecommend:
    def test_returns_list(self, fitted_item_cf: ItemBasedCF) -> None:
        result = fitted_item_cf.recommend(0, n=5)
        assert isinstance(result, list)

    def test_returns_tuples_of_int_and_float(self, fitted_item_cf: ItemBasedCF) -> None:
        result = fitted_item_cf.recommend(0, n=5)
        for movie_idx, score in result:
            assert isinstance(movie_idx, int)
            assert isinstance(score, float)

    def test_does_not_recommend_already_rated_movies(
        self, fitted_item_cf: ItemBasedCF, sample_matrix: csr_matrix
    ) -> None:
        user_idx = 0
        already_rated = set(sample_matrix[user_idx].nonzero()[1])
        recommendations = fitted_item_cf.recommend(user_idx, n=10)
        recommended_ids = {mid for mid, _ in recommendations}
        assert recommended_ids.isdisjoint(already_rated)

    def test_results_are_sorted_descending(self, fitted_item_cf: ItemBasedCF) -> None:
        result = fitted_item_cf.recommend(0, n=10)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_respects_n_limit(self, fitted_item_cf: ItemBasedCF) -> None:
        result = fitted_item_cf.recommend(0, n=2)
        assert len(result) <= 2

    def test_raises_when_not_fitted(self) -> None:
        model = ItemBasedCF(k=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.recommend(0, n=5)


# ─── Shared Interface ─────────────────────────────────────────────────────────

class TestSharedInterface:
    def test_both_models_expose_fit(self) -> None:
        for cls in [UserBasedCF, ItemBasedCF]:
            assert hasattr(cls, "fit")

    def test_both_models_expose_recommend(self) -> None:
        for cls in [UserBasedCF, ItemBasedCF]:
            assert hasattr(cls, "recommend")

    def test_both_models_expose_is_fitted(self) -> None:
        for cls in [UserBasedCF, ItemBasedCF]:
            assert hasattr(cls, "is_fitted")

    def test_recommend_output_format_is_identical(
        self,
        fitted_user_cf: UserBasedCF,
        fitted_item_cf: ItemBasedCF
    ) -> None:
        user_recs = fitted_user_cf.recommend(0, n=3)
        item_recs = fitted_item_cf.recommend(0, n=3)
        for recs in [user_recs, item_recs]:
            for movie_idx, score in recs:
                assert isinstance(movie_idx, int)
                assert isinstance(score, float)