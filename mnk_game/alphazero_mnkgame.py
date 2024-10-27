import time
import math
import torch
import numpy as np

from .mnk_bot_base import MnkGameBotBase
from .alphazero_net import AlphaZeroNet
from board_state.board import to_board
from board_state.mnk_state import MnkState


class AlphaZeroMnkGame(MnkGameBotBase):
    def __init__(self, m, n, k, max_thinking_time, batch, exploration_const,
                 exp_dir, device, debug, **kwargs):
        self.max_thinking_time = max_thinking_time
        self.batch = batch
        self.c = exploration_const
        self.m, self.n, self.k = m, n, k
        self.device = device
        self.debug = debug

    def init_net(self, net=None):
        if net is not None:
            self.net = net
            self.net.to(self.device)
        else:
            self.net = AlphaZeroNet(self.m, self.n, self.k)
            self.net.init_weights()
            self.net.to(self.device)
        self.net.eval()
        return self.net

    def update_tree(self, last_move):
        try:
            if self.debug:
                print("Inheriting previous tree root...")
            self.root = self.root.children[last_move]
            return True
        except KeyError:
            if self.debug:
                print("Moves not found in previous tree. Initializing new tree...")
            return False

    @staticmethod
    def score(node, c):
        # puct
        return node.r / (1 + node.n) + c * node.prior * math.sqrt(node.parent.n) / (1 + node.n)

    def bitboard_to_tensor(self, bb):
        board = to_board(bb, self.m, self.n)
        tensor = torch.tensor(board).float().to(self.device)
        return tensor.unsqueeze(0)

    def solve(self, board, turn, moves):
        start = time.time()
        if len(moves) == 0 or self.root is None:
            if self.debug:
                print("Initializing new tree...")
            self.root = MnkState(board, turn, "blah", None, None)
        else:
            if not self.update_tree(moves[-1]):
                self.root = MnkState(board, turn, "blah", None, None)

        while time.time()-start < self.max_thinking_time:
            self.loop()
        return self.get_results()

    def get_results(self):
        policy = np.zeros((self.m * self.n,), dtype=np.float32)
        for node in self.root.children.values():
            i, j = node.last_move
            policy[i*self.n + j] = node.n
        assert policy.sum() > 0, "No backpropagation found. Please increase max_thinking_time."
        policy /= policy.sum()
        move = np.argmax(policy)
        i, j = move // self.n, move % self.n
        return (i, j), policy

    @torch.no_grad()
    def predict(self, board, turn, _moves):
        policy, value = self.net(self.bitboard_to_tensor(board.get_board()) * turn)
        policy = policy[0]
        self.last_predict_value = value[0]
        possible_move = set(board.get_possible_pos())
        best_move = None
        max_prob = -1.0
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) not in possible_move:
                    continue
                move = i * self.n + j
                if policy[move] > max_prob:
                    max_prob = policy[move]
                    best_move = (i, j)
        return best_move

    def selection(self):
        node = self.root
        while not node.is_leaf():
            selected_node = max(
                node.children.values(), key=lambda child: self.score(child, self.c))
            node = selected_node
        return node

    def expansion(self, node):
        res = node.board.check_endgame()
        if res:
            return res
        states = node.get_next_states()
        if not states:
            return 0.5
        policy, value = self.net(
            self.bitboard_to_tensor(node.board.get_board()) * node.turn
        )
        policy = policy[0]
        value = value[0]
        for state in states:
            i, j = state.last_move
            state.prior = policy[i*self.n + j].item()
        return value.item()

    def backpropagation(self, node, value):
        while node is not None:
            node.n += 1
            node.r += value
            value = 1-value
            node = node.parent

    def loop(self):
        node = self.selection()
        value = self.expansion(node)
        self.backpropagation(node, value)

