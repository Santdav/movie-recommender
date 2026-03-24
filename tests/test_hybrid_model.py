import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


from models.hybrid_model import HybridModel
from src.mf_scratch import MatrixFactorizationSGD
from src.mf_svd import MatrixFactorizationSVD
from models.content_model import ContentModel


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_matrix() -> csr_matrix:
    """
    4 users x 4 movies with clear cluster structure.
    Users 0 and 1 like movies 0 and 1.
    Users 2 and 3 like movies 2 and 3.
    """
    data = np.array([5, 4, 5, 4, 5, 4, 4, 5], dtype=np.float32)
    row  = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    col  = np.array([0, 1, 0, 1, 2, 3, 2, 3])
    return csr_matrix((data, (row, col)), shape=(4, 4))


@pytest.fixture(scope="module")
def movies_df() -> pd.DataFrame:
    return pd.DataFrame({
        "movieId":     [10, 20, 30, 40],
        "title_clean": ["Movie A", "Movie B", "Movie C", "Movie D"],
        "genres": [
            ["Action", "Adventure"],
            ["Action", "Adventure"],
            ["Drama", "Romance"],
            ["Drama", "Romance"],
        ]
    })


@pytest.fixture(scope="module")
def movie_ids() -> np.ndarray:
    # raw movieIds aligned to matrix columns
    return np.array([10, 20, 30, 40])


@pytest.fixture(scope="module")
def fitted_mf_sgd(sample_matrix: csr_matrix) -> MatrixFactorizationSGD:
    return MatrixFactorizationSGD(k=4, n_epochs=50, random_state=42).fit(sample_matrix)


@pytest.fixture(scope="module")
def fitted_mf_svd(sample_matrix: csr_matrix) -> MatrixFactorizationSVD:
    return MatrixFactorizationSVD(k=2, random_state=42).fit(sample_matrix)


@pytest.fixture(scope="module")
def fitted_content(movies_df: pd.DataFrame) -> ContentModel:
    return ContentModel(rating_threshold=4.0).fit(movies_df)


@pytest.fixture(scope="module")
def fitted_hybrid_sgd(
    fitted_mf_sgd: MatrixFactorizationSGD,
    fitted_content: ContentModel,
    sample_matrix: csr_matrix,
    movie_ids: np.ndarray
) -> HybridModel:
    return HybridModel(alpha=0.7).fit(
        fitted_mf_sgd, fitted_content, sample_matrix, movie_ids
    )


@pytest.fixture(scope="module")
def fitted_hybrid_svd(
    fitted_mf_svd: MatrixFactorizationSVD,
    fitted_content: ContentModel,
    sample_matrix: csr_matrix,
    movie_ids: np.ndarray
) -> HybridModel:
    return HybridModel(alpha=0.7).fit(
        fitted_mf_svd, fitted_content, sample_matrix, movie_ids
    )


# ─── Initialization ──────────────────────────────────────────────────────────

class TestInit:
    def test_valid_alpha_accepted(self) -> None:
        model = HybridModel(alpha=0.5)
        assert model.alpha == 0.5

    def test_alpha_zero_accepted(self) -> None:
        model = HybridModel(alpha=0.0)
        assert model.alpha == 0.0

    def test_alpha_one_accepted(self) -> None:
        model = HybridModel(alpha=1.0)
        assert model.alpha == 1.0

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            HybridModel(alpha=1.5)

    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            HybridModel(alpha=-0.1)

    def test_not_fitted_before_fit(self) -> None:
        assert not HybridModel().is_fitted


# ─── Fitting ─────────────────────────────────────────────────────────────────

class TestFit:
    def test_is_fitted_after_fit(self, fitted_hybrid_sgd: HybridModel) -> None:
        assert fitted_hybrid_sgd.is_fitted

    def test_fit_returns_self(
        self,
        fitted_mf_sgd: MatrixFactorizationSGD,
        fitted_content: ContentModel,
        sample_matrix: csr_matrix,
        movie_ids: np.ndarray
    ) -> None:
        model = HybridModel(alpha=0.7)
        result = model.fit(fitted_mf_sgd, fitted_content, sample_matrix, movie_ids)
        assert result is model

    def test_raises_on_unfitted_mf_model(
        self,
        fitted_content: ContentModel,
        sample_matrix: csr_matrix,
        movie_ids: np.ndarray
    ) -> None:
        unfitted_mf = MatrixFactorizationSGD()
        with pytest.raises(RuntimeError, match="MF model must be fitted"):
            HybridModel().fit(unfitted_mf, fitted_content, sample_matrix, movie_ids)

    def test_raises_on_unfitted_content_model(
        self,
        fitted_mf_sgd: MatrixFactorizationSGD,
        sample_matrix: csr_matrix,
        movie_ids: np.ndarray
    ) -> None:
        unfitted_content = ContentModel()
        with pytest.raises(RuntimeError, match="Content model must be fitted"):
            HybridModel().fit(fitted_mf_sgd, unfitted_content, sample_matrix, movie_ids)

    def test_works_with_svd_model(self, fitted_hybrid_svd: HybridModel) -> None:
        assert fitted_hybrid_svd.is_fitted


# ─── Recommend ───────────────────────────────────────────────────────────────

class TestRecommend:
    def test_returns_list(self, fitted_hybrid_sgd: HybridModel) -> None:
        result = fitted_hybrid_sgd.recommend(0, n=3)
        assert isinstance(result, list)

    def test_returns_tuples_of_int_and_float(
        self, fitted_hybrid_sgd: HybridModel
    ) -> None:
        result = fitted_hybrid_sgd.recommend(0, n=3)
        for movie_id, score in result:
            assert isinstance(movie_id, int)
            assert isinstance(score, float)

    def test_respects_n_limit(self, fitted_hybrid_sgd: HybridModel) -> None:
        result = fitted_hybrid_sgd.recommend(0, n=2)
        assert len(result) <= 2

    def test_results_sorted_descending(self, fitted_hybrid_sgd: HybridModel) -> None:
        result = fitted_hybrid_sgd.recommend(0, n=10)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_does_not_recommend_already_rated(
        self, fitted_hybrid_sgd: HybridModel, sample_matrix: csr_matrix, movie_ids: np.ndarray
    ) -> None:
        user_idx = 0
        rated_indices = set(sample_matrix[user_idx].nonzero()[1])
        rated_movie_ids = {int(movie_ids[i]) for i in rated_indices}

        result = fitted_hybrid_sgd.recommend(user_idx, n=10)
        recommended_ids = {mid for mid, _ in result}
        assert recommended_ids.isdisjoint(rated_movie_ids)

    def test_returns_raw_movie_ids_not_indices(
        self, fitted_hybrid_sgd: HybridModel, movie_ids: np.ndarray
    ) -> None:
        result = fitted_hybrid_sgd.recommend(0, n=10)
        returned_ids = {mid for mid, _ in result}
        assert returned_ids.issubset(set(movie_ids.tolist()))

    def test_raises_when_not_fitted(self) -> None:
        model = HybridModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.recommend(0, n=5)

    def test_alpha_zero_uses_only_content(
        self,
        fitted_mf_sgd: MatrixFactorizationSGD,
        fitted_content: ContentModel,
        sample_matrix: csr_matrix,
        movie_ids: np.ndarray
    ) -> None:
        model = HybridModel(alpha=0.0).fit(
            fitted_mf_sgd, fitted_content, sample_matrix, movie_ids
        )
        result = model.recommend(0, n=3)
        assert isinstance(result, list)

    def test_alpha_one_uses_only_mf(
        self,
        fitted_mf_sgd: MatrixFactorizationSGD,
        fitted_content: ContentModel,
        sample_matrix: csr_matrix,
        movie_ids: np.ndarray
    ) -> None:
        model = HybridModel(alpha=1.0).fit(
            fitted_mf_sgd, fitted_content, sample_matrix, movie_ids
        )
        result = model.recommend(0, n=3)
        assert isinstance(result, list)

    def test_svd_hybrid_produces_recommendations(
        self, fitted_hybrid_svd: HybridModel
    ) -> None:
        result = fitted_hybrid_svd.recommend(0, n=3)
        assert len(result) > 0