import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from models.content_model import ContentModel


class HybridModel:
    """
    Weighted blend of a matrix factorization model and a content-based model.
    Final score: alpha * mf_score + (1 - alpha) * content_score

    The MF model handles collaborative signal (what similar users liked).
    The content model handles item similarity (what is similar to what the user liked).
    Blending both produces more robust recommendations than either alone.
    """

    def __init__(self, alpha: float = 0.7) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
        self.alpha = alpha

        self._mf_model = None
        self._content_model: ContentModel | None = None
        self._matrix: csr_matrix | None = None
        self._movie_ids: np.ndarray | None = None

    def fit(
        self,
        mf_model,
        content_model: ContentModel,
        matrix: csr_matrix,
        movie_ids: np.ndarray
    ) -> "HybridModel":
        """
        Attaches already-fitted MF and content models.
        Both models must be fitted before passing to hybrid.

        Args:
            mf_model:      fitted MatrixFactorizationSGD or MatrixFactorizationSVD
            content_model: fitted ContentModel
            matrix:        user-item training matrix from matrix_builder
            movie_ids:     array of raw movieIds aligned to matrix columns (from encoder)
        """
        if not mf_model.is_fitted:
            raise RuntimeError("MF model must be fitted before passing to HybridModel.")
        if not content_model.is_fitted:
            raise RuntimeError("Content model must be fitted before passing to HybridModel.")

        self._mf_model = mf_model
        self._content_model = content_model
        self._matrix = matrix
        self._movie_ids = movie_ids
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10
    ) -> list[tuple[int, float]]:
        """
        Returns top N recommendations for a user by blending MF and content scores.

        Args:
            user_idx: encoded user index
            n:        number of recommendations to return

        Returns:
            list of (movieId, blended_score) tuples sorted descending
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        n_items = self._matrix.shape[1]
        rated_indices = set(self._matrix[user_idx].nonzero()[1])
        rated_mask = np.zeros(n_items, dtype=bool)
        rated_mask[list(rated_indices)] = True

        # MF scores for all unseen items
        mf_scores = self._get_mf_scores(user_idx, rated_mask, n_items)

        # content scores for all unseen items
        content_scores = self._get_content_scores(user_idx, rated_indices, n_items)

        # blend
        blended = self.alpha * mf_scores + (1 - self.alpha) * content_scores
        blended[rated_mask] = -np.inf

        top_indices = np.argsort(blended)[::-1][:n]
        return [
            (int(self._movie_ids[i]), float(blended[i]))
            for i in top_indices
            if blended[i] > -np.inf
        ]

    def _get_mf_scores(
        self,
        user_idx: int,
        rated_mask: np.ndarray,
        n_items: int
    ) -> np.ndarray:
        scores = self._mf_model._predictions[user_idx].copy() \
            if hasattr(self._mf_model, "_predictions") \
            else self._mf_model._mu \
                + self._mf_model._b_u[user_idx] \
                + self._mf_model._b_i \
                + self._mf_model._P[user_idx] @ self._mf_model._Q.T

        scores = np.clip(scores, 1.0, 5.0)
        # normalize to [0, 1] for fair blending
        scores = (scores - 1.0) / 4.0
        return scores

    def _get_content_scores(
        self,
        user_idx: int,
        rated_indices: set[int],
        n_items: int
    ) -> np.ndarray:
        rated_movie_ids = [
            int(self._movie_ids[i]) for i in rated_indices
        ]
        rated_ratings = [
            float(self._matrix[user_idx, i])
            for i in rated_indices
        ]

        # get content recommendations and map back to item index space
        content_recs = self._content_model.recommend(
            rated_movie_ids, rated_ratings, n=n_items
        )

        movie_id_to_score = {mid: score for mid, score in content_recs}
        movie_id_to_idx = {
            int(mid): idx for idx, mid in enumerate(self._movie_ids)
        }

        scores = np.zeros(n_items, dtype=np.float32)
        for movie_id, score in movie_id_to_score.items():
            if movie_id in movie_id_to_idx:
                scores[movie_id_to_idx[movie_id]] = score

        return scores

    @property
    def is_fitted(self) -> bool:
        return (
            self._mf_model is not None
            and self._content_model is not None
            and self._matrix is not None
        )