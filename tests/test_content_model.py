import pytest
import numpy as np
import pandas as pd


from src.content_model import ContentModel


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def movies_df() -> pd.DataFrame:
    """
    6 movies with clear genre clusters.
    Movies 1, 2, 3 are Action/Adventure.
    Movies 4, 5, 6 are Drama/Romance.
    """
    return pd.DataFrame({
        "movieId":     [1, 2, 3, 4, 5, 6],
        "title_clean": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E", "Movie F"],
        "genres": [
            ["Action", "Adventure"],
            ["Action", "Adventure"],
            ["Action", "Adventure"],
            ["Drama", "Romance"],
            ["Drama", "Romance"],
            ["Drama", "Romance"],
        ]
    })


@pytest.fixture(scope="module")
def genome_scores_df() -> pd.DataFrame:
    """
    Minimal genome: 2 tags, clear separation between clusters.
    Tag 1 scores high for action movies, tag 2 scores high for drama movies.
    """
    rows = []
    for movie_id in [1, 2, 3]:
        rows.append({"movieId": movie_id, "tagId": 1, "relevance": 0.9})
        rows.append({"movieId": movie_id, "tagId": 2, "relevance": 0.1})
    for movie_id in [4, 5, 6]:
        rows.append({"movieId": movie_id, "tagId": 1, "relevance": 0.1})
        rows.append({"movieId": movie_id, "tagId": 2, "relevance": 0.9})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_genre_only(movies_df: pd.DataFrame) -> ContentModel:
    return ContentModel(rating_threshold=4.0).fit(movies_df)


@pytest.fixture(scope="module")
def fitted_with_genome(
    movies_df: pd.DataFrame,
    genome_scores_df: pd.DataFrame
) -> ContentModel:
    return ContentModel(rating_threshold=4.0).fit(movies_df, genome_scores_df)


# ─── Fitting — genres only ────────────────────────────────────────────────────

class TestFitGenreOnly:
    def test_is_fitted_after_fit(self, fitted_genre_only: ContentModel) -> None:
        assert fitted_genre_only.is_fitted

    def test_n_movies_correct(
        self, fitted_genre_only: ContentModel, movies_df: pd.DataFrame
    ) -> None:
        assert fitted_genre_only.n_movies == len(movies_df)

    def test_movie_features_shape(
        self, fitted_genre_only: ContentModel, movies_df: pd.DataFrame
    ) -> None:
        assert fitted_genre_only._movie_features.shape[0] == len(movies_df)

    def test_movie_ids_aligned(
        self, fitted_genre_only: ContentModel, movies_df: pd.DataFrame
    ) -> None:
        assert set(fitted_genre_only._movie_ids) == set(movies_df["movieId"].values)

    def test_feature_vectors_are_unit_normalized(
        self, fitted_genre_only: ContentModel
    ) -> None:
        norms = np.linalg.norm(fitted_genre_only._movie_features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_similarity_matrix_shape(self, fitted_genre_only: ContentModel) -> None:
        n = fitted_genre_only.n_movies
        assert fitted_genre_only._similarity.shape == (n, n)

    def test_similarity_matrix_is_symmetric(self, fitted_genre_only: ContentModel) -> None:
        sim = fitted_genre_only._similarity
        np.testing.assert_allclose(sim, sim.T, atol=1e-5)

    def test_similarity_diagonal_is_one(self, fitted_genre_only: ContentModel) -> None:
        diag = np.diag(fitted_genre_only._similarity)
        np.testing.assert_allclose(diag, 1.0, atol=1e-5)

    def test_fit_returns_self(self, movies_df: pd.DataFrame) -> None:
        model = ContentModel()
        result = model.fit(movies_df)
        assert result is model

    def test_not_fitted_before_fit(self) -> None:
        assert not ContentModel().is_fitted


# ─── Fitting — with genome ────────────────────────────────────────────────────

class TestFitWithGenome:
    def test_is_fitted_after_fit(self, fitted_with_genome: ContentModel) -> None:
        assert fitted_with_genome.is_fitted

    def test_feature_vectors_are_unit_normalized(
        self, fitted_with_genome: ContentModel
    ) -> None:
        norms = np.linalg.norm(fitted_with_genome._movie_features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_genome_increases_feature_dimensions(
        self,
        fitted_genre_only: ContentModel,
        fitted_with_genome: ContentModel
    ) -> None:
        assert (
            fitted_with_genome._movie_features.shape[1]
            > fitted_genre_only._movie_features.shape[1]
        )

    def test_cluster_similarity_higher_within_than_across(
        self, fitted_with_genome: ContentModel
    ) -> None:
        # movies 1 and 2 are both action — should be more similar than 1 and 4
        idx1 = fitted_with_genome._movie_id_to_idx[1]
        idx2 = fitted_with_genome._movie_id_to_idx[2]
        idx4 = fitted_with_genome._movie_id_to_idx[4]

        sim_within = fitted_with_genome._similarity[idx1, idx2]
        sim_across = fitted_with_genome._similarity[idx1, idx4]
        assert sim_within > sim_across


# ─── similar_movies ───────────────────────────────────────────────────────────

class TestSimilarMovies:
    def test_returns_list(self, fitted_genre_only: ContentModel) -> None:
        result = fitted_genre_only.similar_movies(1, n=3)
        assert isinstance(result, list)

    def test_returns_tuples_of_int_and_float(
        self, fitted_genre_only: ContentModel
    ) -> None:
        result = fitted_genre_only.similar_movies(1, n=3)
        for movie_id, score in result:
            assert isinstance(movie_id, int)
            assert isinstance(score, float)

    def test_does_not_include_query_movie(
        self, fitted_genre_only: ContentModel
    ) -> None:
        result = fitted_genre_only.similar_movies(1, n=10)
        returned_ids = {mid for mid, _ in result}
        assert 1 not in returned_ids

    def test_results_sorted_descending(self, fitted_genre_only: ContentModel) -> None:
        result = fitted_genre_only.similar_movies(1, n=10)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_respects_n_limit(self, fitted_genre_only: ContentModel) -> None:
        result = fitted_genre_only.similar_movies(1, n=2)
        assert len(result) <= 2

    def test_same_genre_movies_ranked_higher(
        self, fitted_genre_only: ContentModel
    ) -> None:
        # movies 2 and 3 share genres with movie 1
        # movies 4, 5, 6 do not
        result = fitted_genre_only.similar_movies(1, n=5)
        top_ids = {mid for mid, _ in result[:2]}
        assert top_ids.issubset({2, 3})

    def test_returns_empty_for_unknown_movie(
        self, fitted_genre_only: ContentModel
    ) -> None:
        result = fitted_genre_only.similar_movies(999, n=5)
        assert result == []

    def test_raises_when_not_fitted(self) -> None:
        model = ContentModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.similar_movies(1, n=5)


# ─── recommend ───────────────────────────────────────────────────────────────

class TestRecommend:
    def test_returns_list(self, fitted_genre_only: ContentModel) -> None:
        result = fitted_genre_only.recommend([1], [5.0], n=3)
        assert isinstance(result, list)

    def test_returns_tuples_of_int_and_float(
        self, fitted_genre_only: ContentModel
    ) -> None:
        result = fitted_genre_only.recommend([1], [5.0], n=3)
        for movie_id, score in result:
            assert isinstance(movie_id, int)
            assert isinstance(score, float)

    def test_does_not_recommend_already_rated(
        self, fitted_genre_only: ContentModel
    ) -> None:
        rated_ids = [1, 2]
        result = fitted_genre_only.recommend(rated_ids, [5.0, 4.0], n=10)
        returned_ids = {mid for mid, _ in result}
        assert returned_ids.isdisjoint(set(rated_ids))

    def test_results_sorted_descending(self, fitted_genre_only: ContentModel) -> None:
        result = fitted_genre_only.recommend([1], [5.0], n=10)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_respects_n_limit(self, fitted_genre_only: ContentModel) -> None:
        result = fitted_genre_only.recommend([1], [5.0], n=2)
        assert len(result) <= 2

    def test_recommends_same_genre_movies(
        self, fitted_genre_only: ContentModel
    ) -> None:
        # user liked movie 1 (action) — should recommend movies 2 and 3 (also action)
        result = fitted_genre_only.recommend([1], [5.0], n=2)
        top_ids = {mid for mid, _ in result}
        assert top_ids.issubset({2, 3})

    def test_low_rated_movies_excluded_from_profile(
        self, fitted_genre_only: ContentModel
    ) -> None:
        # user rated movie 1 high and movie 4 low
        # profile should reflect movie 1 taste (action) not movie 4 (drama)
        result = fitted_genre_only.recommend([1, 4], [5.0, 1.0], n=2)
        top_ids = {mid for mid, _ in result}
        assert top_ids.issubset({2, 3})

    def test_returns_empty_when_no_liked_movies(
        self, fitted_genre_only: ContentModel
    ) -> None:
        # all ratings below threshold
        result = fitted_genre_only.recommend([1, 2], [2.0, 1.0], n=5)
        assert result == []

    def test_raises_when_not_fitted(self) -> None:
        model = ContentModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.recommend([1], [5.0], n=5)