import time
import numpy as np
import cython
from .board_algorithms import check_board, get_possible_pos
from typing import Tuple
from utils.mixin import PerfMonitorMixin


class MnkBoard(PerfMonitorMixin):
    def __init__(self, m: int, n: int, k: int, board_copy=None) -> None:
        # MAGIC NUMBERS: 1 and 2 are player symbols, 0 is empty cell
        self.m = m  # board width
        self.n = n  # board height
        self.k = k  # k-in-a-row stones for final win
        self.board = None  # game board
        if type(board_copy) == np.ndarray and \
                board_copy.shape == (self.n, self.m):
            self.board = board_copy.copy()
        else:
            self.reset_board()

    def duplicate(self):
        return MnkBoard(self.m, self.n, self.k, self.board)

    def reset_board(self):
        self.board = np.zeros((self.m, self.n), dtype=np.uint8)

    def get_possible_pos(self):
        # old inefficient python code:
        # i, j = np.where(self.board == 0)
        # return np.stack([j, i]).T
        # nEw EfFiCiEnT cYtHoN cOdE:
        return get_possible_pos(self.board)

    def put(self, turn: int, position: Tuple[int, int], display=True):
        assert turn == 1 or turn == 2, "Invalid player: %i" % turn
        self.board[position[1]][position[0]] = turn
        if display:
            print("%s played (%i, %i)" % (
                    "Player" if turn == 1 else "Bot", position[0], position[1]
                ))

    # why are we still here???
    def check_valid(self):
        unique, counts = np.unique(self.board, return_counts=True)
        count_dict = dict(zip(unique, counts))
        if abs(count_dict[1]-count_dict[2]) > 1:
            return False
        return True

    def check_endgame(self):
        start = time.time()
        res = check_board(self.board, self.k)
        # res = self.check_board()
        self.update_perf("check_board", time.time()-start)
        return res

    def check_board(self):
        # inefficient python function for comparing speed
        board = self.board
        board_flip = np.fliplr(board)
        h, w = board.shape[:2]
        i = 0
        res = 0
        # loops through rows:
        for i in range(h):
            res = self.check_line(board[i])
            if res:
                return res
        # loops through columns:
        for i in range(w):
            res = self.check_line(board[:, i])
            if res:
                return res
        # loops through diagonals:
        max_offset = min(h, w) - self.k
        for i in range(-max_offset, max_offset+1):
            res = self.check_line(board.diagonal(i).copy())
            if res: 
                return res
            res = self.check_line(board_flip.diagonal(i).copy())
            if res: 
                return res
        return 0

    def check_line(self, line):
        # inefficient python function for comparing speed
        s = 0
        consecutive = 0
        i = 0
        line_len = line.shape[0]
        while i < line_len:
            if line[i] == 0:
                consecutive = 0
                s = 0
            else:
                if s != line[i]:
                    s = line[i]
                    consecutive = 1
                else:
                    consecutive += 1
                    if consecutive == self.k:
                        return s
            i += 1
        return 0


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
