import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, normalize


class ContentModel:
    """
    Content-based filtering using movie genres and tag genome relevance scores.
    Builds a feature vector for each movie by combining:
        - One-hot encoded genres
        - Tag genome relevance scores (1,128 tags from ML-25M)
    Recommends movies similar to those a user has already rated highly,
    based on cosine similarity between movie feature vectors.
    """

    def __init__(self, rating_threshold: float = 4.0) -> None:
        self.rating_threshold = rating_threshold  # minimum rating to consider a movie "liked"

        self._movie_features: np.ndarray | None = None  # shape: (n_movies, n_features)
        self._movie_ids: np.ndarray | None = None       # movieId values aligned to feature rows
        self._similarity: np.ndarray | None = None      # shape: (n_movies, n_movies)
        self._movie_id_to_idx: dict[int, int] = {}

    def fit(
        self,
        movies_df: pd.DataFrame,
        genome_scores_df: pd.DataFrame | None = None
    ) -> "ContentModel":
        """
        Builds movie feature vectors from genres and optionally tag genome scores.

        Args:
            movies_df:        DataFrame from load_movies() with movieId and genres columns
            genome_scores_df: DataFrame from load_genome_scores() with movieId, tagId, relevance
                              Pass None to use genres only.
        """
        genre_features = self._build_genre_features(movies_df)

        if genome_scores_df is not None:
            genome_features = self._build_genome_features(movies_df, genome_scores_df)
            # only keep movies that appear in both genres and genome
            common_ids = np.intersect1d(movies_df["movieId"].values, genome_features.index.values)
            genre_mask = movies_df["movieId"].isin(common_ids).values
            genre_features = genre_features[genre_mask]
            genome_matrix = genome_features.loc[common_ids].values.astype(np.float32)
            self._movie_ids = common_ids
            features = np.hstack([genre_features, genome_matrix])
        else:
            self._movie_ids = movies_df["movieId"].values
            features = genre_features

        # L2 normalize so cosine similarity is just a dot product
        self._movie_features = normalize(features, norm="l2").astype(np.float32)
        self._movie_id_to_idx = {mid: idx for idx, mid in enumerate(self._movie_ids)}
        self._similarity = self._movie_features @ self._movie_features.T

        print(f"Content model fitted — {len(self._movie_ids)} movies, {features.shape[1]} features")
        return self

    def _build_genre_features(self, movies_df: pd.DataFrame) -> np.ndarray:
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(movies_df["genres"]).astype(np.float32)
        return genre_matrix

    def _build_genome_features(
        self,
        movies_df: pd.DataFrame,
        genome_scores_df: pd.DataFrame
    ) -> pd.DataFrame:
        # pivot genome scores to (movieId x tagId) matrix
        genome_pivot = genome_scores_df.pivot(
            index="movieId", columns="tagId", values="relevance"
        ).fillna(0.0)
        return genome_pivot

    def similar_movies(
        self,
        movie_id: int,
        n: int = 10
    ) -> list[tuple[int, float]]:
        """
        Returns the n most similar movies to the given movie_id.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if movie_id not in self._movie_id_to_idx:
            return []

        idx = self._movie_id_to_idx[movie_id]
        sim_scores = self._similarity[idx].copy()
        sim_scores[idx] = -1  # exclude the movie itself

        top_indices = np.argsort(sim_scores)[::-1][:n + 1]
        return [
            (int(self._movie_ids[i]), float(sim_scores[i]))
            for i in top_indices
            if self._movie_ids[i] != movie_id
        ][:n]

    def recommend(
        self,
        rated_movie_ids: list[int],
        rated_scores: list[float],
        n: int = 10
    ) -> list[tuple[int, float]]:
        """
        Recommends movies based on a user's rated movies.
        Builds a user taste profile by averaging feature vectors of
        highly rated movies, then finds the most similar unseen movies.

        Args:
            rated_movie_ids: list of movieIds the user has rated
            rated_scores:    corresponding rating values
            n:               number of recommendations to return
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        liked_ids = [
            mid for mid, score in zip(rated_movie_ids, rated_scores)
            if score >= self.rating_threshold and mid in self._movie_id_to_idx
        ]

        if not liked_ids:
            return []

        # build user taste profile as mean of liked movie feature vectors
        liked_indices = [self._movie_id_to_idx[mid] for mid in liked_ids]
        user_profile = self._movie_features[liked_indices].mean(axis=0)
        user_profile = user_profile / (np.linalg.norm(user_profile) + 1e-10)

        # score all movies against the user profile
        scores = self._movie_features @ user_profile

        # exclude already rated movies
        rated_set = set(rated_movie_ids)
        results = [
            (int(self._movie_ids[i]), float(scores[i]))
            for i in range(len(self._movie_ids))
            if self._movie_ids[i] not in rated_set
        ]

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    @property
    def is_fitted(self) -> bool:
        return self._movie_features is not None

    @property
    def n_movies(self) -> int:
        return len(self._movie_ids) if self._movie_ids is not None else 0