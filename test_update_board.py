import numpy as np
from apputil import update_board, play_random_game


def test_blinker_oscillator():
    # vertical blinker should become horizontal after one update
    b = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    expected = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    out = update_board(b)
    assert np.array_equal(out, expected)


def test_block_still_life():
    blk = np.array([[1, 1], [1, 1]])
    out = update_board(blk)
    assert np.array_equal(out, blk)


def test_play_random_game_returns_shape_and_binary():
    final = play_random_game(max_steps=5)
    assert final.shape == (10, 10)
    # values should be 0 or 1 only
    uniques = np.unique(final)
    assert set(uniques).issubset({0, 1})


def test_play_random_game_on_stable_board():
    blk = np.array([[1, 1], [1, 1]])
    out = play_random_game(blk, max_steps=10)
    assert np.array_equal(out, blk)
