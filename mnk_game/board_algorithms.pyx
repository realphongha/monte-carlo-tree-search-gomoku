# cython: infer_types=True
import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.uint8


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def check_line(np.uint8_t[:] line, int num):  # int[:] - memoryviews for numpy array
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
