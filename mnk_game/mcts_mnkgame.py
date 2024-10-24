import time
import random
import multiprocessing
from typing import Tuple

import numpy as np

from mcts.mcts import MonteCarloTreeSearchMixin
from .mnk_bot_base import MnkGameBotBase
from board_state.mnk_state import MnkState, rollout
from board_state.mnk_board import MnkBoard


class MonteCarloTreeSearchMnkGame(MonteCarloTreeSearchMixin, MnkGameBotBase):
    def __init__(self, max_thinking_time, max_rollout, processes, policy,
                 exploration_const, num_simulations) -> None:
        super().__init__(max_thinking_time)
        self.max_rollout = max_rollout
        self.processes = processes
        self.pool = multiprocessing.Pool(self.processes)
        self.policy = policy
        self.c = exploration_const
        self.num_simulations = num_simulations
        self.root = None

    def update_tree(self, two_last_moves):
        try:
            print("Inheriting previous tree root...")
            m1, m2 = two_last_moves
            self.root = self.root.children[m1].children[m2]
            return True
        except KeyError:
            print("Moves not found in previous tree. Initializing new tree...")
            return False

    def solve(self, board: MnkBoard, turn: int, moves) -> Tuple[int, int]:
        start = time.time()
        if len(moves) < 2 or self.root is None:
            print("Initializing new tree...")
            self.root = MnkState(board, turn, self.policy, None, None)
        else:
            if not self.update_tree(moves[-2:]):
                self.root = MnkState(board, turn, self.policy, None, None)

        while time.time()-start < self.max_thinking_time and \
                self.total_rollout < self.max_rollout:
            self.loop()
        return self.get_results()

    def get_move_winrate(self, move):
        child = self.root.children.get(move, None)
        return self.ucb(child, 0) if child.n != 0 else None

    def get_results(self):
        best_child = None
        if self.total_rollout > 0 and len(self.root.children.values()) > 0:
            children = []
            for child in self.root.children.values():
                children.append((child, self.ucb(child, 0)))
            top_k = 5 if len(children) >= 5 else len(children)
            children.sort(key=lambda child: -child[1])
            print("\nTop %i moves:" % top_k)
            for child, score in children[:5]:
                print("Move:", child.last_move, "- score: %.4f - w: %i - n: %i" %
                    (score, child.r, child.n)
                )
            best_child = children[0][0]
        print("Played %i rollouts!" % self.rollout_count)
        print("Total: %i rollouts (inherited from previous trees)!" %
            self.total_rollout)
        return best_child.last_move if (best_child and self.total_rollout > 0) else (-1, -1)

    def selection(self):
        node = self.root
        while not node.is_leaf():
            selected_node = max(
                node.children.values(), key=lambda child: self.ucb(child, self.c))
            node = selected_node
        return node

    def choosing_policy(self, states):
        return random.choice(states)

    def expansion(self, node):
        if node.board.check_endgame() == 0 and node.board.get_possible_pos():
            next_states = node.next_states()
            next_states = [state for state in next_states if state.n == 0]
            old_states = []
            for state in next_states:
                if state.last_move not in node.children:
                    node.children[state.last_move] = state
                old_states.append(node.children[state.last_move])
            if old_states:
                node = self.choosing_policy(old_states)
        return node

    def simulation(self, node):
        self.rollout_count += self.num_simulations
        self.total_rollout += self.num_simulations
        if self.num_simulations == 1:
            return [node.rollout()]
        args = [(node.board.duplicate(), node.turn) for _ in range(self.num_simulations)]
        return self.pool.starmap(rollout, args)

    def backpropagation(self, node, winner, times=1):
        # node.turn means the next player
        if winner == node.turn:
            reward = 0
        elif winner == 0:  # a draw
            reward = 0.5
        else:
            reward = 1
        while node is not None:
            node.n += times
            node.r += reward * times
            node = node.parent
            reward = 1-reward

