import os
import joblib

PROCESSED_DIR = "data/processed"
USER_ENCODER_PATH = os.path.join(PROCESSED_DIR, "user_encoder.joblib")
MOVIE_ENCODER_PATH = os.path.join(PROCESSED_DIR, "movie_encoder.joblib")


def build_encoders(ratings_df):
    user_encoder = {uid: idx for idx, uid in enumerate(sorted(ratings_df["userId"].unique()))}
    movie_encoder = {mid: idx for idx, mid in enumerate(sorted(ratings_df["movieId"].unique()))}

    user_decoder = {idx: uid for uid, idx in user_encoder.items()}
    movie_decoder = {idx: mid for mid, idx in movie_encoder.items()}

    return user_encoder, movie_encoder, user_decoder, movie_decoder


def apply_encoders(ratings_df, user_encoder, movie_encoder):
    df = ratings_df.copy()
    df["user_idx"] = df["userId"].map(user_encoder)
    df["movie_idx"] = df["movieId"].map(movie_encoder)

    missing_users = df["user_idx"].isna().sum()
    missing_movies = df["movie_idx"].isna().sum()
    if missing_users > 0 or missing_movies > 0:
        raise ValueError(
            f"Encoding failed: {missing_users} unknown users, {missing_movies} unknown movies. "
            "Ensure encoders were built from the full dataset before splitting."
        )

    df["user_idx"] = df["user_idx"].astype(int)
    df["movie_idx"] = df["movie_idx"].astype(int)
    return df


def save_encoders(user_encoder, movie_encoder, user_decoder, movie_decoder):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump({"encoder": user_encoder, "decoder": user_decoder}, USER_ENCODER_PATH)
    joblib.dump({"encoder": movie_encoder, "decoder": movie_decoder}, MOVIE_ENCODER_PATH)
    print(f"Encoders saved to {PROCESSED_DIR}/")


def load_encoders():
    user_data = joblib.load(USER_ENCODER_PATH)
    movie_data = joblib.load(MOVIE_ENCODER_PATH)
    return (
        user_data["encoder"],
        movie_data["encoder"],
        user_data["decoder"],
        movie_data["decoder"],
    )


def encode(ratings_df):
    user_encoder, movie_encoder, user_decoder, movie_decoder = build_encoders(ratings_df)
    encoded_df = apply_encoders(ratings_df, user_encoder, movie_encoder)
    save_encoders(user_encoder, movie_encoder, user_decoder, movie_decoder)

    print(f"Users  : {len(user_encoder)}")
    print(f"Movies : {len(movie_encoder)}")

    return encoded_df, user_encoder, movie_encoder, user_decoder, movie_decoder