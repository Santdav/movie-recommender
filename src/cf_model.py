import numpy as np
from scipy.sparse import csr_matrix


# ─── User-Based CF ────────────────────────────────────────────────────────────

class UserBasedCF:
    """
    User-based collaborative filtering using cosine similarity.
    Recommends items liked by users most similar to the target user.
    """

    def __init__(self, k: int = 20) -> None:
        self.k = k
        self._matrix: csr_matrix | None = None
        self._similarity: np.ndarray | None = None
        self._n_users: int = 0
        self._n_movies: int = 0

    def fit(self, matrix: csr_matrix) -> "UserBasedCF":
        self._matrix = matrix
        self._n_users, self._n_movies = matrix.shape
        self._similarity = self._compute_cosine_similarity(matrix)
        return self

    def _compute_cosine_similarity(self, matrix: csr_matrix) -> np.ndarray:
        dense = matrix.toarray().astype(np.float32)

        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # avoid division by zero for users with no ratings

        normalized = dense / norms
        return normalized @ normalized.T  # shape: (n_users, n_users)

    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        if self._matrix is None or self._similarity is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        sim_scores = self._similarity[user_idx].copy()
        sim_scores[user_idx] = -1  # exclude the target user from their own neighborhood

        neighbor_indices = np.argsort(sim_scores)[::-1][:self.k]
        neighbor_ratings = self._matrix[neighbor_indices, movie_idx].toarray().flatten()
        neighbor_sims = sim_scores[neighbor_indices]

        rated_mask = neighbor_ratings > 0
        if rated_mask.sum() == 0:
            return 0.0

        weighted_sum = np.dot(neighbor_sims[rated_mask], neighbor_ratings[rated_mask])
        sim_sum = np.abs(neighbor_sims[rated_mask]).sum()

        if sim_sum == 0:
            return 0.0

        return float(weighted_sum / sim_sum)

    def recommend(
        self,
        user_idx: int,
        n: int = 10
    ) -> list[tuple[int, float]]:
        if self._matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        already_rated = set(self._matrix[user_idx].nonzero()[1])
        unseen_movies = [i for i in range(self._n_movies) if i not in already_rated]

        scores = [
            (movie_idx, self.predict_rating(user_idx, movie_idx))
            for movie_idx in unseen_movies
        ]

        scores = [(mid, s) for mid, s in scores if s > 0.0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    @property
    def is_fitted(self) -> bool:
        return self._matrix is not None and self._similarity is not None


# ─── Item-Based CF ────────────────────────────────────────────────────────────

class ItemBasedCF:
    """
    Item-based collaborative filtering using adjusted cosine similarity.
    Recommends items similar to those the target user has already rated highly.
    """

    def __init__(self, k: int = 20) -> None:
        self.k = k
        self._matrix: csr_matrix | None = None
        self._similarity: np.ndarray | None = None
        self._n_users: int = 0
        self._n_movies: int = 0

    def fit(self, matrix: csr_matrix) -> "ItemBasedCF":
        self._matrix = matrix
        self._n_users, self._n_movies = matrix.shape
        self._similarity = self._compute_adjusted_cosine_similarity(matrix)
        return self

    def _compute_adjusted_cosine_similarity(self, matrix: csr_matrix) -> np.ndarray:
        dense = matrix.toarray().astype(np.float32)

        # subtract each user's mean rating to account for rating bias
        user_means = np.true_divide(
            dense.sum(axis=1),
            (dense > 0).sum(axis=1),
            where=(dense > 0).sum(axis=1) > 0
        )
        user_means = np.nan_to_num(user_means)

        adjusted = dense.copy()
        for u in range(self._n_users):
            rated_mask = dense[u] > 0
            adjusted[u, rated_mask] -= user_means[u]

        # compute cosine similarity on item vectors (columns)
        item_matrix = adjusted.T  # shape: (n_movies, n_users)
        norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10

        normalized = item_matrix / norms
        return normalized @ normalized.T  # shape: (n_movies, n_movies)

    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        if self._matrix is None or self._similarity is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        user_ratings = self._matrix[user_idx].toarray().flatten()
        rated_indices = np.where(user_ratings > 0)[0]

        if len(rated_indices) == 0:
            return 0.0

        sim_scores = self._similarity[movie_idx][rated_indices]
        ratings = user_ratings[rated_indices]

        top_k_mask = np.argsort(np.abs(sim_scores))[::-1][:self.k]
        sim_scores_k = sim_scores[top_k_mask]
        ratings_k = ratings[top_k_mask]

        sim_sum = np.abs(sim_scores_k).sum()
        if sim_sum == 0:
            return 0.0

        return float(np.dot(sim_scores_k, ratings_k) / sim_sum)

    def recommend(
        self,
        user_idx: int,
        n: int = 10
    ) -> list[tuple[int, float]]:
        if self._matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        already_rated = set(self._matrix[user_idx].nonzero()[1])
        unseen_movies = [i for i in range(self._n_movies) if i not in already_rated]

        scores = [
            (movie_idx, self.predict_rating(user_idx, movie_idx))
            for movie_idx in unseen_movies
        ]

        scores = [(mid, s) for mid, s in scores if s > 0.0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    @property
    def is_fitted(self) -> bool:
        return self._matrix is not None and self._similarity is not None