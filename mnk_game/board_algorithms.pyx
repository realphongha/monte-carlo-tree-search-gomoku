# cython: infer_types=True
import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.uint8


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int check_line(np.uint8_t[:] line, int num):  # int[:] - memoryviews for numpy array
    # check if there's a symbol appears `num` times consecutively
    cdef np.uint8_t s = 0
    cdef int consecutive = 0
    cdef Py_ssize_t i = 0  # Py_ssize_t for indexing
    cdef Py_ssize_t line_len = line.shape[0]
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
                if consecutive == num:
                    return s
        i += 1
    return 0


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def check_board(np.ndarray[np.uint8_t, ndim=2] board_np, int num):
    # checks if game is ended or not, and who won
    cdef np.uint8_t[:, :] board = board_np
    cdef np.ndarray[np.uint8_t, ndim=2] board_np_flip = np.fliplr(board_np)
    cdef Py_ssize_t h = board.shape[0]
    cdef Py_ssize_t w = board.shape[1]
    cdef Py_ssize_t i = 0
    cdef int res = 0
    # loops through rows:
    for i in range(h):
        res = check_line(board[i], num)
        if res:
            return res
    # loops through columns:
    for i in range(w):
        res = check_line(board[:, i], num)
        if res:
            return res
    # loops through diagonals:
    cdef Py_ssize_t max_offset = min(h, w) - num
    for i in range(-max_offset, max_offset+1):
        res = check_line(board_np.diagonal(i).copy(), num)
        if res: 
            return res
        res = check_line(board_np_flip.diagonal(i).copy(), num)
        if res: 
            return res
    return 0


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int check_board_cdef(np.ndarray[np.uint8_t, ndim=2] board_np, int num,
        int last_i=-1, int last_j=-1):
    # checks if game is ended or not, and who won
    cdef np.uint8_t[:, :] board = board_np
    cdef np.ndarray[np.uint8_t, ndim=2] board_np_flip = np.fliplr(board_np)
    cdef Py_ssize_t h = board.shape[0]
    cdef Py_ssize_t w = board.shape[1]
    cdef Py_ssize_t i = 0
    cdef int res = 0
    if last_i == -1:
        # loops through rows:
        for i in range(h):
            res = check_line(board[i], num)
            if res:
                return res
    else:
        # only checks last move:
        res = check_line(board[last_j], num)
        if res:
            return res
    if last_i == -1:
        # loops through columns:
        for i in range(w):
            res = check_line(board[:, i], num)
            if res:
                return res
    else:
        # only checks last move:
        res = check_line(board[:, last_i], num)
        if res:
            return res
    cdef Py_ssize_t max_offset = min(h, w) - num
    cdef int os
    if last_i == -1:
        # loops through diagonals:
        for i in range(-max_offset, max_offset+1):
            res = check_line(board_np.diagonal(i).copy(), num)
            if res: 
                return res
            res = check_line(board_np_flip.diagonal(i).copy(), num)
            if res: 
                return res
    else:
        # only checks last move:
        os = last_i-last_j
        if -max_offset <= os <= max_offset:
            res = check_line(board_np.diagonal(os).copy(), num)
            if res:
                return res
        os = w-1-last_i-last_j
        if -max_offset <= os <= max_offset:
            res = check_line(board_np_flip.diagonal(os).copy(), num)
            if res: 
                return res
    return 0


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_possible_pos(np.uint8_t[:, :] board):
    # finds all possible moves on board
    cdef Py_ssize_t h = board.shape[0]
    cdef Py_ssize_t w = board.shape[1]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    res = []
    for i in range(h):
        for j in range(w):
            if board[i][j] == 0:
                res.append((j, i))
    return res


cdef class MnkBoard:
    cdef:
        public int m
        public int n
        int k
        public np.ndarray board

    def __init__(self, int m, int n, int k, board_copy=None):
        # MAGIC NUMBERS: 1 and 2 are player symbols, 0 is empty cell
        self.m = m  # board width
        self.n = n  # board height
        self.k = k  # k-in-a-row for final win
        if type(board_copy) == np.ndarray and \
                board_copy.shape == (self.n, self.m):
            self.board = board_copy.copy()
        else:
            self.reset_board()

    def duplicate(self):
        return MnkBoard(self.m, self.n, self.k, self.board)

    def reset_board(self):
        self.board = np.zeros((self.n, self.m), dtype=np.uint8)

    def get_possible_pos(self):
        return get_possible_pos(self.board)

    def put(self, int turn, position, display=True):
        assert turn == 1 or turn == 2, "Invalid player: %i" % turn
        self.board[position[1]][position[0]] = turn
        if display:
            print("%s played (%i, %i)" % (
                    "Player" if turn == 1 else "Bot", position[0], position[1]
                ))

    def check_endgame(self, int last_i=-1, int last_j=-1):
        return check_board_cdef(self.board, self.k, last_i, last_j)

    cdef get_nearby_pos(self, pos):
        # gets <= 8 nearby pos:
        cdef Py_ssize_t i, j
        ret_pos = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not i == j == 0 and 0 <= pos[0]+i < self.m and 0 <= pos[1]+j < self.n:
                    ret_pos.append((pos[0]+i, pos[1]+j))
        return ret_pos

    cdef int get_dist_to_nearest_symbol(self, pos, int stop=2):
        cdef Py_ssize_t i, j
        cdef np.uint8_t[:, :] visited = np.zeros((self.n, self.m), dtype=np.uint8)
        queue = [pos]
        visited[pos[1], pos[0]] = 1
        while queue:
            s = queue.pop(0)
            if self.board[s[1], s[0]] != 0:
                return visited[pos[1], pos[0]]-1
            if visited[s[1], s[0]] > stop + 1:
                continue
            for i, j in self.get_nearby_pos(s):
                if visited[j, i] == 0:
                    queue.append((i, j))
                    visited[j, i] = visited[s[1], s[0]] + 1
        return stop + 1

    cpdef bint is_near_a_symbol(self, pos):
        cdef Py_ssize_t i, j
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not i == j == 0 and 0 <= pos[0]+i < self.m and 0 <= pos[1]+j < self.n and self.board[pos[1]+j][pos[0]+i] != 0:
                    return True
        return False 
