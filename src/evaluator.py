import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def evaluate_user_based_cf(
    model,
    test_df: pd.DataFrame,
    sample_size: int = 2000
) -> float:
    """
    Evaluates UserBasedCF or ItemBasedCF on a sample of the test set.
    Sampling is necessary because predict_rating is O(n_users) per call.

    Args:
        model:       fitted UserBasedCF or ItemBasedCF instance
        test_df:     test DataFrame with user_idx, movie_idx, rating columns
        sample_size: number of test ratings to evaluate on
    """
    sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

    actual = sample["rating"].values.astype(np.float32)
    predicted = np.array([
        model.predict_rating(int(row.user_idx), int(row.movie_idx))
        for row in sample.itertuples()
    ], dtype=np.float32)

    return _rmse(actual, predicted)


def evaluate_mf_sgd(
    model,
    test_df: pd.DataFrame
) -> float:
    """
    Evaluates MatrixFactorizationSGD on the full test set.
    predict_rating is a fast lookup so no sampling needed.
    """
    actual = test_df["rating"].values.astype(np.float32)
    predicted = np.array([
        model.predict_rating(int(row.user_idx), int(row.movie_idx))
        for row in test_df.itertuples()
    ], dtype=np.float32)

    return _rmse(actual, predicted)


def evaluate_mf_svd(
    model,
    test_df: pd.DataFrame
) -> float:
    """
    Evaluates MatrixFactorizationSVD on the full test set.
    Predictions are precomputed at fit time so this is just an array lookup.
    """
    actual = test_df["rating"].values.astype(np.float32)
    predicted = np.array([
        model.predict_rating(int(row.user_idx), int(row.movie_idx))
        for row in test_df.itertuples()
    ], dtype=np.float32)

    return _rmse(actual, predicted)


def evaluate_all(
    models: dict,
    test_df: pd.DataFrame,
    cf_sample_size: int = 2000
) -> pd.DataFrame:
    """
    Evaluates all models and returns a comparison DataFrame sorted by RMSE.

    Args:
        models: dict mapping model name to fitted model instance.
                Supported keys: "user_cf", "item_cf", "mf_sgd", "mf_svd"
        test_df:         test DataFrame with user_idx, movie_idx, rating columns
        cf_sample_size:  sample size for CF models

    Returns:
        DataFrame with columns: model, rmse
    """
    cf_models = {"user_cf", "item_cf"}
    sgd_models = {"mf_sgd"}
    svd_models = {"mf_svd"}

    results = []

    for name, model in models.items():
        print(f"Evaluating {name}...")

        if name in cf_models:
            rmse = evaluate_user_based_cf(model, test_df, sample_size=cf_sample_size)
        elif name in sgd_models:
            rmse = evaluate_mf_sgd(model, test_df)
        elif name in svd_models:
            rmse = evaluate_mf_svd(model, test_df)
        else:
            raise ValueError(
                f"Unknown model key '{name}'. "
                "Expected one of: 'user_cf', 'item_cf', 'mf_sgd', 'mf_svd'."
            )

        print(f"  RMSE: {rmse:.4f}")
        results.append({"model": name, "rmse": rmse})

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    return results_df