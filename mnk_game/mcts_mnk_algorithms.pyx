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

    def __init__(self, board, int turn, str policy, last_move, 
            MnkState parent, children=None, int n=0, float r=0.0) -> None:
        self.board = board
        self.turn = turn
        self.policy = policy
        self.last_move = last_move
        self.parent = parent
        self.children = children if children else dict()
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
        states = list()
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
        pos = test_board.get_possible_pos()
        while res == 0 and len(pos) != 0:
            if self.policy == "simple":
                index = random.randrange(0, len(pos))
                # index = simple_rollout_policy(pos)        
            elif self.policy == "prob":
                index = prob_rollout_policy(test_board, pos)
            else:
                raise NotImplementedError("Policy %s is not implemented!" %
                    self.policy)
            i, j = pos[index]
            pos.pop(index)
            test_board.board[j][i] = turn
            turn = 3-turn
            res = test_board.check_endgame(i, j)
        return res


cdef int simple_rollout_policy(pos):
    # completely random moves
    return random.randrange(0, len(pos))


cdef int prob_rollout_policy(board, pos):
    # cells near previously marked cell get greater probability to be chosen
    cdef Py_ssize_t num = len(pos)
    cdef Py_ssize_t i
    cdef np.uint8_t[:] dummy = np.arange(len(pos), dtype=np.uint8)
    nearby = [i for i, p in enumerate(pos) if board.get_dist_to_nearest_symbol(p) <= 2]
    cdef int num_nearby = len(nearby)
    cdef int num_faraway = num - num_nearby
    if num_nearby == num:
        prob = [1.0 / num] * len(pos)
    elif num_faraway == num:
        prob = [1.0 / num] * len(pos)
    else:
        prob = list()
        for i in range(num):
            prob.append(0.9 / num_nearby if i in nearby else (0.1 / num_faraway))
    index = np.random.choice(dummy, p=prob)
    return index
