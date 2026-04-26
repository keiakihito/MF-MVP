"""
MovieLens 1M dataset loader.

Expected raw files (download and place in data/raw/movielens_1m/):
    ratings.dat  — UserID::MovieID::Rating::Timestamp
    users.dat    — UserID::Gender::Age::Occupation::Zip-code
    movies.dat   — MovieID::Title::Genres  (Genres separated by |)

Download:
    https://grouplens.org/datasets/movielens/1m/
    Unzip into data/raw/movielens_1m/

Public API
----------
load_ratings(data_dir) -> pd.DataFrame
    Columns: [user_id (int), movie_id (int), rating (float), timestamp (int)]

load_users(data_dir) -> pd.DataFrame
    Columns: [user_id (int), gender (str), age (int), occupation (int), zip (str)]

load_movies(data_dir) -> pd.DataFrame
    Columns: [movie_id (int), title (str), genres (list[str])]
"""

from __future__ import annotations

import os
import pandas as pd

_SEP = "::"
_ENCODING = "latin-1"  # MovieLens 1M uses latin-1, not utf-8


def load_ratings(data_dir: str) -> pd.DataFrame:
    """
    Load ratings.dat into a DataFrame.

    Args:
        data_dir: Path to the directory containing ratings.dat.

    Returns:
        DataFrame with columns [user_id, movie_id, rating, timestamp].

    Raises:
        FileNotFoundError: if ratings.dat is not found in data_dir.
    """
    path = os.path.join(data_dir, "ratings.dat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ratings.dat not found in {data_dir}")
    df = pd.read_csv(path, sep=_SEP, header=None, engine="python", encoding=_ENCODING,
                     names=["user_id", "movie_id", "rating", "timestamp"])
    df["user_id"]   = df["user_id"].astype(int)
    df["movie_id"]  = df["movie_id"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def load_users(data_dir: str) -> pd.DataFrame:
    """
    Load users.dat into a DataFrame.

    Args:
        data_dir: Path to the directory containing users.dat.

    Returns:
        DataFrame with columns [user_id, gender, age, occupation, zip].

    Raises:
        FileNotFoundError: if users.dat is not found in data_dir.
    """
    path = os.path.join(data_dir, "users.dat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"users.dat not found in {data_dir}")
    df = pd.read_csv(path, sep=_SEP, header=None, engine="python", encoding=_ENCODING,
                     names=["user_id", "gender", "age", "occupation", "zip"])
    df["user_id"]    = df["user_id"].astype(int)
    df["age"]        = df["age"].astype(int)
    df["occupation"] = df["occupation"].astype(int)
    df["gender"]     = df["gender"].astype(object)
    df["zip"]        = df["zip"].astype(object)
    return df


def load_movies(data_dir: str) -> pd.DataFrame:
    """
    Load movies.dat into a DataFrame.

    Genres are stored as a pipe-separated string (e.g. "Action|Adventure|Sci-Fi").
    This function parses them into a Python list of strings.

    Args:
        data_dir: Path to the directory containing movies.dat.

    Returns:
        DataFrame with columns [movie_id, title, genres]
        where genres is a list[str].

    Raises:
        FileNotFoundError: if movies.dat is not found in data_dir.
    """
    path = os.path.join(data_dir, "movies.dat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"movies.dat not found in {data_dir}")
    df = pd.read_csv(path, sep=_SEP, header=None, engine="python", encoding=_ENCODING,
                     names=["movie_id", "title", "genres"])
    df["movie_id"] = df["movie_id"].astype(int)
    df["genres"]   = df["genres"].apply(lambda g: g.split("|"))
    return df
