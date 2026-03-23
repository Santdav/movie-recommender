import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class MatrixFactorizationSVD:
    """
    Matrix factorization using truncated SVD from scipy.
    Decomposes the mean-centered user-item matrix into:
        R_centered ≈ U × Σ × Vᵀ
    Predicted rating: U[u] @ diag(Σ) @ V[i] + user_mean[u]

    Key difference from SGD: this is a closed-form solution, not iterative.
    No learning rate or epochs required — one call solves the factorization.
    """

    def __init__(self, k: int = 50, random_state: int = 42) -> None:
        self.k = k
        self.random_state = random_state

        self._U: np.ndarray | None = None        # user vectors  (n_users x k)
        self._sigma: np.ndarray | None = None    # singular values (k,)
        self._Vt: np.ndarray | None = None       # item vectors  (k x n_items)
        self._user_means: np.ndarray | None = None
        self._predictions: np.ndarray | None = None  # full reconstructed matrix

    def fit(self, matrix: csr_matrix) -> "MatrixFactorizationSVD":
        dense = matrix.toarray().astype(np.float32)

        # mean-center by user to remove rating bias before decomposition
        self._user_means = np.true_divide(
            dense.sum(axis=1),
            (dense > 0).sum(axis=1),
            where=(dense > 0).sum(axis=1) > 0
        ).astype(np.float32)
        self._user_means = np.nan_to_num(self._user_means)

        centered = dense.copy()
        for u in range(dense.shape[0]):
            rated = dense[u] > 0
            centered[u, rated] -= self._user_means[u]

        # k must be strictly less than min(n_users, n_items)
        k = min(self.k, min(centered.shape) - 1)

        self._U, self._sigma, self._Vt = svds(
            csr_matrix(centered),
            k=k,
            random_state=self.random_state
        )

        # reconstruct the full predicted matrix once at fit time
        # this avoids recomputing it on every predict/recommend call
        reconstructed = self._U @ np.diag(self._sigma) @ self._Vt
        self._predictions = reconstructed + self._user_means.reshape(-1, 1)
        self._predictions = np.clip(self._predictions, 1.0, 5.0)

        print(f"SVD complete — k={k}, matrix shape: {self._predictions.shape}")
        return self

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return float(self._predictions[user_idx, item_idx])

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        rated_mask: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        scores = self._predictions[user_idx].copy()

        if rated_mask is not None:
            scores[rated_mask] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > -np.inf]

    @property
    def is_fitted(self) -> bool:
        return self._predictions is not None