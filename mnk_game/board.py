import numpy as np
from .board_algorithms import MnkBoard


if __name__ == "__main__":
    from tqdm import tqdm
    k = 5
    # board = [[2, 1, 2, 0],
    #          [0, 2, 0, 2],
    #          [2, 1, 0, 0],
    #          [1, 2, 0, 0]]
    board = np.zeros((15, 15))
    board = np.array(board).astype(np.uint8)
    mnk_board = MnkBoard(board.shape[1], board.shape[0], k, board)
    for _ in tqdm(range(10)):
        res = mnk_board.check_endgame()
    print(res)
    mnk_board.get_perf("check_board", True)
