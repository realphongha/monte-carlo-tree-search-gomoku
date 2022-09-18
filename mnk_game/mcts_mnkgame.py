import time
import math
import random
from typing import Tuple
from mcts.mcts import MonteCarloTreeSearchMixin
from .mnk_bot_base import MnkGameBotBase
from .board import MnkBoard
from utils import constants


class MnkState:
    def __init__(self, board: MnkBoard, turn, policy, last_move, parent, 
            children=None, n=0, r=0) -> None:
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
        for k in range(pos.shape[0]):
            i, j = pos[k]
            new_board = self.board.duplicate()
            new_board.board[j][i] = 3-self.turn
            new_state = MnkState(new_board, 3-self.turn, self.policy, 
                (i, j), self)
            states.append(new_state)
        return states

    @staticmethod
    def simple_rollout_policy(board):
        # completely random moves
        pos = board.get_possible_pos()
        if pos.shape[0] == 0:
            return -1, -1
        index = random.randrange(0, pos.shape[0])
        return pos[index]

    def prob_rollout_policy(self):
        # cells near previously marked cell get greater probability to be chosen
        pass

    def rollout(self):
        test_board = self.board.duplicate()
        turn = self.turn
        res = test_board.check_end_game()
        while res == 0:
            if self.policy == "simple":
                i, j = self.simple_rollout_policy(test_board)
            else:
                raise NotImplementedError("Policy %s is not implemented!" %
                    self.policy)
            if i == j == -1:
                return 0
            test_board.board[j][i] = turn
            turn = 3-turn
            res = test_board.check_end_game()
        return res


class MonteCarloTreeSearchMnkGame(MonteCarloTreeSearchMixin, MnkGameBotBase):
    def __init__(self, thinking_time, processes, policy) -> None:
        super().__init__(thinking_time, processes)
        self.policy = policy

    def solve(self, board: MnkBoard, turn: int) -> Tuple[int, int]:
        start = time.time()
        self.root = MnkState(board, turn, self.policy, None, None)
        loop_times = 0
        while time.time()-start < self.thinking_time:
            node = self.selection()
            node = self.expansion(node)
            winner = self.simulation(node)
            self.backpropagation(node, winner)
            loop_times += 1
        best_child = max(self.root.children.values(), key=self.score)
        print("Score:")
        for move, child in self.root.children.items():
            print(move, self.score(child), child.r, child.n)
        print("Loop %i times!" % loop_times)
        return best_child.last_move

    def selection(self):
        node = self.root
        while not node.is_leaf():
            selected_node = max(node.children.values(), key=lambda child: 
                self.ucb(
                    child.r, child.n, constants.EXPLORATION_CONST, node.n
                ))
            node = selected_node
        return node
        
    def choosing_policy(self, states):
        return random.choice(states)

    def expansion(self, node):
        if node.board.check_end_game() == 0:
            next_states = node.next_states()
            next_states = [state for state in next_states if state.n == 0]
            old_states = list()
            for state in next_states:
                if state.last_move not in node.children:
                    node.children[state.last_move] = state
                old_states.append(node.children[state.last_move])
            if old_states:
                node = self.choosing_policy(old_states)
        return node

    def simulation(self, node):
        return node.rollout()

    def backpropagation(self, node, winner):
        if winner == node.turn:
            reward = 1
        elif winner == 0:  # a draw
            reward = 0.5
        else: 
            reward = 0
        while node != self.root:
            node.n += 1
            node.r += reward
            node = node.parent
            reward = 1-reward
        # updates the root as well:
        node.n += 1
        node.r += reward

    @staticmethod
    def score(node):
        return node.r/node.n if node.n != 0 else -math.inf
