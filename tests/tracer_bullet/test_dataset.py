"""
Tests for dataset.py — MovieLens 1M loader contracts.

Tests use pytest's tmp_path fixture to create synthetic .dat files in the
MovieLens format. No real MovieLens download is required.

TDD: contracts defined here; dataset.py raises NotImplementedError until implemented.
"""

import os
import pytest

from tracer_bullet.dataset import load_ratings, load_users, load_movies

pytestmark = pytest.mark.unit


# ── Helpers to write synthetic .dat files ─────────────────────────────────────

def write_ratings_dat(path: str) -> None:
    """Write a minimal synthetic ratings.dat in MovieLens format."""
    with open(path, "w") as f:
        f.write("1::1193::5::978300760\n")
        f.write("1::661::3::978302109\n")
        f.write("2::3408::4::978300275\n")


def write_users_dat(path: str) -> None:
    """Write a minimal synthetic users.dat in MovieLens format."""
    with open(path, "w") as f:
        f.write("1::F::1::10::48067\n")
        f.write("2::M::56::16::70072\n")


def write_movies_dat(path: str, encoding: str = "latin-1") -> None:
    """Write a minimal synthetic movies.dat in MovieLens format."""
    with open(path, "w", encoding=encoding) as f:
        f.write("1::Toy Story (1995)::Animation|Children's|Comedy\n")
        f.write("2::Jumanji (1995)::Adventure|Children's|Fantasy\n")


# ── load_ratings ──────────────────────────────────────────────────────────────

def test_load_ratings_columns(tmp_path):
    write_ratings_dat(tmp_path / "ratings.dat")
    df = load_ratings(str(tmp_path))
    assert set(df.columns) == {"user_id", "movie_id", "rating", "timestamp"}


def test_load_ratings_row_count(tmp_path):
    write_ratings_dat(tmp_path / "ratings.dat")
    df = load_ratings(str(tmp_path))
    assert len(df) == 3


def test_load_ratings_dtypes(tmp_path):
    write_ratings_dat(tmp_path / "ratings.dat")
    df = load_ratings(str(tmp_path))
    assert df["user_id"].dtype  == int
    assert df["movie_id"].dtype == int
    assert df["timestamp"].dtype == int
    # rating may be int or float — just confirm it's numeric
    assert df["rating"].dtype.kind in ("i", "f")


def test_load_ratings_values(tmp_path):
    write_ratings_dat(tmp_path / "ratings.dat")
    df = load_ratings(str(tmp_path))
    row = df[df["user_id"] == 1].iloc[0]
    assert row["movie_id"] == 1193
    assert row["rating"] == 5


def test_load_ratings_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ratings(str(tmp_path))  # no ratings.dat written


# ── load_users ────────────────────────────────────────────────────────────────

def test_load_users_columns(tmp_path):
    write_users_dat(tmp_path / "users.dat")
    df = load_users(str(tmp_path))
    assert set(df.columns) == {"user_id", "gender", "age", "occupation", "zip"}


def test_load_users_row_count(tmp_path):
    write_users_dat(tmp_path / "users.dat")
    df = load_users(str(tmp_path))
    assert len(df) == 2


def test_load_users_dtypes(tmp_path):
    write_users_dat(tmp_path / "users.dat")
    df = load_users(str(tmp_path))
    assert df["user_id"].dtype    == int
    assert df["age"].dtype        == int
    assert df["occupation"].dtype == int
    assert df["gender"].dtype     == object  # string


def test_load_users_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_users(str(tmp_path))


# ── load_movies ───────────────────────────────────────────────────────────────

def test_load_movies_columns(tmp_path):
    write_movies_dat(tmp_path / "movies.dat")
    df = load_movies(str(tmp_path))
    assert set(df.columns) == {"movie_id", "title", "genres"}


def test_load_movies_genres_is_list(tmp_path):
    write_movies_dat(tmp_path / "movies.dat")
    df = load_movies(str(tmp_path))
    assert isinstance(df["genres"].iloc[0], list)


def test_load_movies_genres_parsed_correctly(tmp_path):
    write_movies_dat(tmp_path / "movies.dat")
    df = load_movies(str(tmp_path))
    toy_story = df[df["movie_id"] == 1].iloc[0]
    assert "Animation" in toy_story["genres"]
    assert "Comedy"    in toy_story["genres"]


def test_load_movies_row_count(tmp_path):
    write_movies_dat(tmp_path / "movies.dat")
    df = load_movies(str(tmp_path))
    assert len(df) == 2


def test_load_movies_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_movies(str(tmp_path))
