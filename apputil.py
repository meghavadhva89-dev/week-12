import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import clear_output


def update_board(current_board):
    """
    Perform one Conway's Game of Life step on a binary NumPy array.

    Parameters
    ----------
    current_board : numpy.ndarray
        2D binary array where 1 indicates a live cell and 0 a dead cell.

    Returns
    -------
    numpy.ndarray
        New 2D binary array after one update step.
    """
    board = np.asarray(current_board, dtype=int)

    # pad with zeros so we can compute neighbors for edge cells
    padded = np.pad(board, pad_width=1, mode="constant", constant_values=0)

    # sum the eight neighbors using slicing of the padded array
    neighbors = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )

    # apply the Game of Life rules
    updated = np.zeros_like(board)

    # live cells with 2 or 3 neighbors survive
    updated[(board == 1) & ((neighbors == 2) | (neighbors == 3))] = 1

    # dead cells with exactly 3 neighbors become live
    updated[(board == 0) & (neighbors == 3)] = 1

    return updated


def play_random_game(board=None, max_steps=50):
    """
    Play Conway's Game of Life recursively for a random 10x10 board (or given board).

    Parameters
    ----------
    board : numpy.ndarray or None
        If None, a random 10x10 board is created. Otherwise the provided board is used.
    max_steps : int
        Maximum number of recursive steps to run.

    Returns
    -------
    numpy.ndarray
        Final board when stable or when `max_steps` is exhausted.
    """
    if board is None:
        board = np.random.randint(2, size=(10, 10))

    next_board = update_board(board)

    if np.array_equal(next_board, board) or max_steps <= 1:
        return next_board

    return play_random_game(next_board, max_steps - 1)


def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show `n_steps` of Conway's Game of Life, given the `update_board` function.

    Parameters
    ----------
    game_board : numpy.ndarray
        A binary array representing the initial starting conditions for Conway's Game of Life. In this array, ` represents a "living" cell and 0 represents a "dead" cell.
    n_steps : int, optional
        Number of game steps to run through, by default 10
    pause : float, optional
        Number of seconds to wait between steps, by default 0.5
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # show board
        sns.heatmap(game_board, cmap="plasma", cbar=False, square=True)
        plt.title(f"Board State at Step {step + 1}")
        plt.show()

        # wait for the next step
        if step + 1 < n_steps:
            time.sleep(pause)
