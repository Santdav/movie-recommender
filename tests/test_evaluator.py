import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


from src.evaluator import _rmse, evaluate_user_based_cf, evaluate_mf_sgd, evaluate_mf_svd, evaluate_all
from models.cf_model import UserBasedCF, ItemBasedCF
from src.mf_scratch import MatrixFactorizationSGD
from src.mf_svd import MatrixFactorizationSVD


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_matrix() -> csr_matrix:
    data = np.array([5, 4, 5, 4, 5, 4, 4, 5], dtype=np.float32)
    row  = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    col  = np.array([0, 1, 0, 1, 2, 3, 2, 3])
    return csr_matrix((data, (row, col)), shape=(4, 4))


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_idx":  [0, 1, 2, 3, 0, 1],
        "movie_idx": [0, 1, 2, 3, 2, 3],
        "rating":    [5.0, 4.0, 5.0, 4.0, 3.0, 2.0]
    })


@pytest.fixture(scope="module")
def fitted_user_cf(sample_matrix: csr_matrix) -> UserBasedCF:
    return UserBasedCF(k=2).fit(sample_matrix)


@pytest.fixture(scope="module")
def fitted_item_cf(sample_matrix: csr_matrix) -> ItemBasedCF:
    return ItemBasedCF(k=2).fit(sample_matrix)


@pytest.fixture(scope="module")
def fitted_mf_sgd(sample_matrix: csr_matrix) -> MatrixFactorizationSGD:
    return MatrixFactorizationSGD(k=4, n_epochs=50, random_state=42).fit(sample_matrix)


@pytest.fixture(scope="module")
def fitted_mf_svd(sample_matrix: csr_matrix) -> MatrixFactorizationSVD:
    return MatrixFactorizationSVD(k=2, random_state=42).fit(sample_matrix)


# ─── _rmse ───────────────────────────────────────────────────────────────────

class TestRmse:
    def test_perfect_predictions_give_zero_rmse(self) -> None:
        actual    = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _rmse(actual, predicted) == pytest.approx(0.0, abs=1e-6)

    def test_known_rmse_value(self) -> None:
        actual    = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        # all errors are 1.0, so RMSE = sqrt(mean([1,1,1])) = 1.0
        assert _rmse(actual, predicted) == pytest.approx(1.0, abs=1e-6)

    def test_returns_float(self) -> None:
        result = _rmse(np.array([1.0]), np.array([2.0]))
        assert isinstance(result, float)

    def test_rmse_is_non_negative(self) -> None:
        actual    = np.array([5.0, 3.0, 1.0])
        predicted = np.array([1.0, 5.0, 3.0])
        assert _rmse(actual, predicted) >= 0.0

    def test_rmse_is_symmetric(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert _rmse(a, b) == pytest.approx(_rmse(b, a), abs=1e-6)


# ─── evaluate_user_based_cf ──────────────────────────────────────────────────

class TestEvaluateUserBasedCF:
    def test_returns_float(
        self, fitted_user_cf: UserBasedCF, test_df: pd.DataFrame
    ) -> None:
        result = evaluate_user_based_cf(fitted_user_cf, test_df)
        assert isinstance(result, float)

    def test_rmse_is_non_negative(
        self, fitted_user_cf: UserBasedCF, test_df: pd.DataFrame
    ) -> None:
        result = evaluate_user_based_cf(fitted_user_cf, test_df)
        assert result >= 0.0

    def test_rmse_in_plausible_range(
        self, fitted_user_cf: UserBasedCF, test_df: pd.DataFrame
    ) -> None:
        result = evaluate_user_based_cf(fitted_user_cf, test_df)
        assert result <= 4.0  # max possible RMSE on a 1-5 scale

    def test_sample_size_respected(
        self, fitted_user_cf: UserBasedCF, test_df: pd.DataFrame
    ) -> None:
        # should not raise even when sample_size > len(test_df)
        result = evaluate_user_based_cf(fitted_user_cf, test_df, sample_size=10000)
        assert isinstance(result, float)

    def test_item_cf_also_works(
        self, fitted_item_cf: ItemBasedCF, test_df: pd.DataFrame
    ) -> None:
        result = evaluate_user_based_cf(fitted_item_cf, test_df)
        assert isinstance(result, float)


# ─── evaluate_mf_sgd ─────────────────────────────────────────────────────────

class TestEvaluateMFSGD:
    def test_returns_float(
        self, fitted_mf_sgd: MatrixFactorizationSGD, test_df: pd.DataFrame
    ) -> None:
        result = evaluate_mf_sgd(fitted_mf_sgd, test_df)
        assert isinstance(result, float)

    def test_rmse_is_non_negative(
        self, fitted_mf_sgd: MatrixFactorizationSGD, test_df: pd.DataFrame
    ) -> None:
        result = evaluate_mf_sgd(fitted_mf_sgd, test_df)
        assert result >= 0.0

    def test_rmse_in_plausible_range(
        self, fitted_mf_sgd: MatrixFactorizationSGD, test_df: pd.DataFrame
    ) -> None:
        result = evaluate_mf_sgd(fitted_mf_sgd, test_df)
        assert result <= 4.0

    def test_evaluates_full_test_set(
        self, fitted_mf_sgd: MatrixFactorizationSGD, test_df: pd.DataFrame
    ) -> None:
        # SGD evaluates all rows, verify no sampling occurs by checking
        # result is deterministic across two calls
        r1 = evaluate_mf_sgd(fitted_mf_sgd, test_df)
        r2 = evaluate_mf_sgd(fitted_mf_sgd, test_df)
        assert r1 == pytest.approx(r2, abs=1e-6)


# ─── evaluate_mf_svd ─────────────────────────────────────────────────────────

class TestEvaluateMFSVD:
    def test_returns_float(
        self, fitted_mf_svd: MatrixFactorizationSVD, test_df: pd.DataFrame
    ) -> None:
        result = evaluate_mf_svd(fitted_mf_svd, test_df)
        assert isinstance(result, float)

    def test_rmse_is_non_negative(
        self, fitted_mf_svd: MatrixFactorizationSVD, test_df: pd.DataFrame
    ) -> None:
        result = evaluate_mf_svd(fitted_mf_svd, test_df)
        assert result >= 0.0

    def test_rmse_in_plausible_range(
        self, fitted_mf_svd: MatrixFactorizationSVD, test_df: pd.DataFrame
    ) -> None:
        result = evaluate_mf_svd(fitted_mf_svd, test_df)
        assert result <= 4.0


# ─── evaluate_all ────────────────────────────────────────────────────────────

class TestEvaluateAll:
    def test_returns_dataframe(
        self,
        fitted_user_cf: UserBasedCF,
        fitted_mf_svd: MatrixFactorizationSVD,
        test_df: pd.DataFrame
    ) -> None:
        models = {"user_cf": fitted_user_cf, "mf_svd": fitted_mf_svd}
        result = evaluate_all(models, test_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_correct_columns(
        self,
        fitted_user_cf: UserBasedCF,
        test_df: pd.DataFrame
    ) -> None:
        result = evaluate_all({"user_cf": fitted_user_cf}, test_df)
        assert list(result.columns) == ["model", "rmse"]

    def test_row_count_matches_models(
        self,
        fitted_user_cf: UserBasedCF,
        fitted_mf_sgd: MatrixFactorizationSGD,
        fitted_mf_svd: MatrixFactorizationSVD,
        test_df: pd.DataFrame
    ) -> None:
        models = {
            "user_cf": fitted_user_cf,
            "mf_sgd":  fitted_mf_sgd,
            "mf_svd":  fitted_mf_svd,
        }
        result = evaluate_all(models, test_df)
        assert len(result) == len(models)

    def test_results_sorted_by_rmse_ascending(
        self,
        fitted_user_cf: UserBasedCF,
        fitted_mf_svd: MatrixFactorizationSVD,
        test_df: pd.DataFrame
    ) -> None:
        models = {"user_cf": fitted_user_cf, "mf_svd": fitted_mf_svd}
        result = evaluate_all(models, test_df)
        rmse_values = result["rmse"].tolist()
        assert rmse_values == sorted(rmse_values)

    def test_all_rmse_values_are_non_negative(
        self,
        fitted_user_cf: UserBasedCF,
        fitted_mf_svd: MatrixFactorizationSVD,
        test_df: pd.DataFrame
    ) -> None:
        models = {"user_cf": fitted_user_cf, "mf_svd": fitted_mf_svd}
        result = evaluate_all(models, test_df)
        assert (result["rmse"] >= 0.0).all()

    def test_raises_on_unknown_model_key(
        self,
        fitted_user_cf: UserBasedCF,
        test_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="Unknown model key"):
            evaluate_all({"unknown_model": fitted_user_cf}, test_df)

    def test_all_four_models_together(
        self,
        fitted_user_cf: UserBasedCF,
        fitted_item_cf: ItemBasedCF,
        fitted_mf_sgd: MatrixFactorizationSGD,
        fitted_mf_svd: MatrixFactorizationSVD,
        test_df: pd.DataFrame
    ) -> None:
        models = {
            "user_cf": fitted_user_cf,
            "item_cf": fitted_item_cf,
            "mf_sgd":  fitted_mf_sgd,
            "mf_svd":  fitted_mf_svd,
        }
        result = evaluate_all(models, test_df)
        assert len(result) == 4
        assert set(result["model"]) == set(models.keys())
        