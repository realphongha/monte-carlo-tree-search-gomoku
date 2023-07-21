# cython: infer_types=True
import random
import numpy as np
cimport numpy as np


cdef class MnkState:
    cdef:
        public board
        public int turn
        str policy
        public last_move
        public MnkState parent
        public dict children
        public int n
        public float r
        public np.ndarray prob 

    def __init__(self, board, int turn, str policy, last_move, 
            MnkState parent, children=None, int n=0, float r=0.0) -> None:
        self.board = board
        self.turn = turn
        self.policy = policy
        self.last_move = last_move
        self.parent = parent
        self.children = children if children else {}
        self.n = n
        self.r = r

    def is_leaf(self):
        if not self.children:
            return True
        for child in self.children.values():
            if child.n == 0:
                return True
        return False

    def next_states(self):
        pos = self.board.get_possible_pos()
        states = []
        for p in pos:
            new_board = self.board.duplicate()
            new_board.board[p[1]][p[0]] = 3-self.turn
            new_state = MnkState(new_board, 3-self.turn, self.policy, 
                p, self)
            states.append(new_state)
        return states

    def merge(self, MnkState other, parent, bint in_place=True):
        # self and other must have similar parents to merge
        if self.last_move != other.last_move:
            return False
        if in_place:
            state = self
            state.n += other.n
            state.r += other.r
        else:
            state = MnkState(self.board.duplicate(), self.turn, self.policy,
                self.last_move, parent, self.children, self.n+other.n, self.r+other.r)
        return state

    def rollout(self):
        test_board = self.board.duplicate()
        cdef int turn = self.turn
        cdef int res = test_board.check_endgame()
        cdef int i, j, index
        cdef list pos = test_board.get_possible_pos()
        cdef np.ndarray near_symbol
        if self.policy == "prob":
            near_symbol = self.get_near_symbol_list(test_board, pos)
        while res == 0 and len(pos) != 0:
            if self.policy == "simple":
                index = random.randrange(0, len(pos))
                i, j = pos[index]
            elif self.policy == "prob":
                index = self.prob_rollout_policy(test_board, pos, near_symbol)
                i, j = pos[index]
                self.mark_near_symbol_list(near_symbol, i, j, test_board.m, test_board.n)
            else:
                raise NotImplementedError("Policy %s is not implemented!" %
                    self.policy)
            pos.pop(index)
            test_board.board[j][i] = turn
            turn = 3-turn
            res = test_board.check_endgame(i, j)
        return res

    cdef np.ndarray get_near_symbol_list(self, board, list pos):
        cdef Py_ssize_t i
        cdef np.ndarray near_symbol = np.zeros((board.n, board.m), dtype=np.uint8)
        for i, p in enumerate(pos):
            if board.is_near_a_symbol(p):
                near_symbol[p[1]][p[0]] = 1
        return near_symbol

    cdef void mark_near_symbol_list(self, near_symbol, int x, int y, int m, int n):
        cdef Py_ssize_t i, j
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not i == j == 0 and 0 <= x+i < m and 0 <= y+j < n:
                    near_symbol[y+j][x+i] = 1

    cdef int prob_rollout_policy(self, board, list pos, near_symbol):
        cdef Py_ssize_t i, j, index
        cdef int x, y
        cdef list choices = []
        for i, p in enumerate(pos):
            if near_symbol[p[1]][p[0]]:
                # 19 is a magic number that means 
                # cells near previously marked cell get 95% to be chosen
                # feel free to change it
                for j in range(19):
                    choices.append(i)
            else:
                choices.append(i)
        return random.choice(choices)
