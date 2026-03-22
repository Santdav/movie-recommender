import pytest
import pandas as pd


from src.splitter import time_based_split, split


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_ratings() -> pd.DataFrame:
    return pd.DataFrame({
        "userId":    [1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
        "movieId":   [10, 20, 10, 30, 20, 40, 10, 20, 30, 40],
        "rating":    [5.0, 3.0, 4.0, 2.0, 5.0, 1.0, 3.0, 4.0, 2.0, 5.0],
        "user_idx":  [0, 0, 1, 1, 2, 2, 3, 3, 3, 3],
        "movie_idx": [0, 1, 0, 2, 1, 3, 0, 1, 2, 3],
        "timestamp": pd.to_datetime([
            "2000-01-01", "2000-02-01", "2000-03-01", "2000-04-01",
            "2000-05-01", "2000-06-01", "2000-07-01", "2000-08-01",
            "2000-09-01", "2000-10-01"
        ])
    })


# ─── time_based_split ─────────────────────────────────────────────────────────

class TestTimeBasedSplit:
    def test_returns_two_dataframes(self, sample_ratings: pd.DataFrame) -> None:
        train, test = time_based_split(sample_ratings)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_correct_total_row_count(self, sample_ratings: pd.DataFrame) -> None:
        train, test = time_based_split(sample_ratings)
        assert len(train) + len(test) == len(sample_ratings)

    def test_approximate_split_ratio(self, sample_ratings: pd.DataFrame) -> None:
        train, test = time_based_split(sample_ratings, test_ratio=0.2)
        actual_ratio = len(test) / len(sample_ratings)
        assert abs(actual_ratio - 0.2) <= 0.1

    def test_train_timestamps_before_test(self, sample_ratings: pd.DataFrame) -> None:
        train, test = time_based_split(sample_ratings)
        assert train["timestamp"].max() <= test["timestamp"].min()

    def test_no_overlap_between_splits(self, sample_ratings: pd.DataFrame) -> None:
        train, test = time_based_split(sample_ratings)
        train_timestamps = set(train["timestamp"])
        test_timestamps = set(test["timestamp"])
        assert train_timestamps.isdisjoint(test_timestamps)

    def test_train_is_sorted_by_timestamp(self, sample_ratings: pd.DataFrame) -> None:
        train, _ = time_based_split(sample_ratings)
        assert train["timestamp"].is_monotonic_increasing

    def test_test_is_sorted_by_timestamp(self, sample_ratings: pd.DataFrame) -> None:
        _, test = time_based_split(sample_ratings)
        assert test["timestamp"].is_monotonic_increasing

    def test_raises_on_non_datetime_timestamp(self) -> None:
        bad_df = pd.DataFrame({
            "userId":    [1, 2],
            "movieId":   [10, 20],
            "rating":    [4.0, 3.0],
            "timestamp": [1000000, 2000000]  # raw unix integers
        })
        with pytest.raises(TypeError, match="timestamp column must be datetime"):
            time_based_split(bad_df)

    def test_train_has_reset_index(self, sample_ratings: pd.DataFrame) -> None:
        train, _ = time_based_split(sample_ratings)
        assert list(train.index) == list(range(len(train)))

    def test_test_has_reset_index(self, sample_ratings: pd.DataFrame) -> None:
        _, test = time_based_split(sample_ratings)
        assert list(test.index) == list(range(len(test)))

    def test_all_columns_are_preserved(self, sample_ratings: pd.DataFrame) -> None:
        train, test = time_based_split(sample_ratings)
        assert list(train.columns) == list(sample_ratings.columns)
        assert list(test.columns) == list(sample_ratings.columns)

    def test_custom_test_ratio(self, sample_ratings: pd.DataFrame) -> None:
        train, test = time_based_split(sample_ratings, test_ratio=0.3)
        actual_ratio = len(test) / len(sample_ratings)
        assert abs(actual_ratio - 0.3) <= 0.1

    def test_does_not_mutate_input(self, sample_ratings: pd.DataFrame) -> None:
        original_len = len(sample_ratings)
        original_cols = list(sample_ratings.columns)
        time_based_split(sample_ratings)
        assert len(sample_ratings) == original_len
        assert list(sample_ratings.columns) == original_cols