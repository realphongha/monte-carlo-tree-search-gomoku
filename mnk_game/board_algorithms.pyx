# cython: infer_types=True
import numpy as np
cimport numpy as np
cimport cython
from gmpy2 cimport *
import_gmpy2()   # needed to initialize the C-API


DTYPE = np.uint8


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef bint check_board_cdef(bb, n, k):
    cdef Py_ssize_t i
    cdef mpz res
    # vertical
    res = bb
    for i in range(1, k):
        res &= (bb >> i)
    if res:
        return True
    
    # horizontal
    res = bb
    for i in range(1, k):
        res &= (bb >> ((n+1)*i))
    if res:
        return True
    
    # diagonal \
    res = bb
    for i in range(1, k):
        res &= (bb >> (n*i))
    if res:
        return True
    
    # diagonal /
    res = bb
    for i in range(1, k):
        res &= (bb >> ((n+2)*i))
    if res:
        return True
    return False


cdef class MnkBoard:
    cdef:
        public int m
        public int n
        int k
        list board

    def __init__(self, int m, int n, int k, board_copy=None):
        # MAGIC NUMBERS: 1 and 2 are player symbols, 0 is empty cell
        self.m = m  # board width
        self.n = n  # board height
        self.k = k  # k-in-a-row for final win
        if type(board_copy) == list:
            self.board = board_copy.copy()
        else:
            self.reset_board()

    def get_board(self):
        # for debug only
        return self.board

    def duplicate(self):
        return MnkBoard(self.m, self.n, self.k, self.board)

    def reset_board(self):
        self.board = [mpz(0), mpz(0)]  # [player1_bitboard, player2_bitboard]

    def get_possible_pos(self):
        cdef int i, j
        cdef list res = []
        for i in range(self.m):
            for j in range(self.n):
                if self.is_empty(i, j):
                    res.append((i, j))
        return res

    def put(self, int turn, position, display=True):
        cdef int i, j 
        cdef mpz bitmask
        # assert turn == 1 or turn == 2, "Invalid player: %i" % turn
        if display:
            print("%s played (%i, %i)" % (
                    "Player" if turn == 1 else "Bot", position[0], position[1]
                ))
        turn -= 1
        i, j = position
        j = self.n - 1 - j 
        bitmask = mpz(1) << (j + i*(self.n+1))
        # assert bitmask >= 0, f"Int overflow: 1 << {j + i*(self.n+1)} = {bitmask}"
        self.board[turn] += bitmask
    
    cdef bint get(self, int turn, int i, int j):
        cdef mpz bitmask
        turn -= 1
        bitmask = mpz(1) << (self.n - 1 - j + i*(self.n+1))
        # assert bitmask >= 0, f"Int overflow: 1 << {self.n - 1 - j + i*(self.n+1)} = {bitmask}"
        return  self.board[turn] & bitmask

    def index(self, int i, int j):
        if self.get(1, i, j):
            return 1
        if self.get(2, i, j):
            return 2
        return 0

    cdef bint is_empty(self, int i, int j):
        return not (self.get(1, i, j) or self.get(2, i, j))

    def check_endgame(self, int last_i=-1, int last_j=-1):
        cdef bint p1, p2
        p1 = check_board_cdef(self.board[0], self.n, self.k)
        if p1:
            return 1
        p2 = check_board_cdef(self.board[1], self.n, self.k)
        if p2:
            return 2
        return 0

    cpdef bint is_near_a_symbol(self, pos):
        cdef Py_ssize_t i, j
        cdef mpz bitmask = mpz(0)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not i == j == 0 and 0 <= pos[0]+i < self.m and 0 <= pos[1]+j < self.n: 
                    bitmask += mpz(1) << (self.n - 1 - pos[1] - j + (pos[0]+i) * (self.n+1))
        return (self.board[0] & bitmask) or (self.board[1] & bitmask)

