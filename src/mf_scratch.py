import numpy as np
from scipy.sparse import csr_matrix


class MatrixFactorizationSGD:
    """
    Matrix factorization trained with Stochastic Gradient Descent.
    Decomposes the user-item rating matrix into two latent factor matrices:
        - P: user factors  (n_users x k)
        - Q: item factors  (n_items x k)
    Predicted rating for user u and item i: P[u] @ Q[i].T + b_u[u] + b_i[i] + mu
    """

    def __init__(
        self,
        k: int = 50,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        random_state: int = 42
    ) -> None:
        self.k = k
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

        self._P: np.ndarray | None = None       # user factors
        self._Q: np.ndarray | None = None       # item factors
        self._b_u: np.ndarray | None = None     # user biases
        self._b_i: np.ndarray | None = None     # item biases
        self._mu: float = 0.0                   # global mean rating
        self._train_loss: list[float] = []

    def fit(self, matrix: csr_matrix) -> "MatrixFactorizationSGD":
        rng = np.random.default_rng(self.random_state)
        n_users, n_items = matrix.shape

        # initialize latent factors with small random values
        scale = 1.0 / np.sqrt(self.k)
        self._P = rng.normal(0, scale, (n_users, self.k)).astype(np.float32)
        self._Q = rng.normal(0, scale, (n_items, self.k)).astype(np.float32)
        self._b_u = np.zeros(n_users, dtype=np.float32)
        self._b_i = np.zeros(n_items, dtype=np.float32)

        # global mean of all known ratings
        self._mu = matrix.data.mean()

        # extract known ratings as (user_idx, item_idx, rating) triples
        cx = matrix.tocoo()
        samples = list(zip(cx.row, cx.col, cx.data))

        for epoch in range(self.n_epochs):
            rng.shuffle(samples)
            epoch_loss = self._run_epoch(samples)
            self._train_loss.append(epoch_loss)
            print(f"Epoch {epoch + 1:>3}/{self.n_epochs}  loss: {epoch_loss:.4f}")

        return self

    def _run_epoch(self, samples: list) -> float:
        total_loss = 0.0

        for u, i, r in samples:
            pred = self._predict_raw(u, i)
            err = r - pred

            # update biases
            self._b_u[u] += self.lr * (err - self.reg * self._b_u[u])
            self._b_i[i] += self.lr * (err - self.reg * self._b_i[i])

            # update latent factors
            p_u = self._P[u].copy()
            self._P[u] += self.lr * (err * self._Q[i] - self.reg * self._P[u])
            self._Q[i] += self.lr * (err * p_u - self.reg * self._Q[i])

            total_loss += err ** 2

        return float(total_loss / len(samples))

    def _predict_raw(self, user_idx: int, item_idx: int) -> float:
        return (
            self._mu
            + self._b_u[user_idx]
            + self._b_i[item_idx]
            + float(self._P[user_idx] @ self._Q[item_idx])
        )

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        raw = self._predict_raw(user_idx, item_idx)
        return float(np.clip(raw, 1.0, 5.0))

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        rated_mask: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        n_items = self._Q.shape[0]

        # compute scores for all items in one vectorized operation
        scores = self._mu + self._b_u[user_idx] + self._b_i + self._P[user_idx] @ self._Q.T
        scores = np.clip(scores, 1.0, 5.0)

        # mask already rated items
        if rated_mask is not None:
            scores[rated_mask] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > -np.inf]

    @property
    def is_fitted(self) -> bool:
        return self._P is not None and self._Q is not None

    @property
    def train_loss(self) -> list[float]:
        return self._train_loss