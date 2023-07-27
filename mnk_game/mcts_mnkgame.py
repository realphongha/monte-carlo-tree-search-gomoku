import time
import random
import multiprocessing
from typing import Tuple

import numpy as np

from mcts.mcts import MonteCarloTreeSearchMixin
from .mnk_bot_base import MnkGameBotBase
from .board import MnkBoard
from .mcts_mnk_algorithms import MnkState


last_tree = None


class MonteCarloTreeSearchMnkGame(MonteCarloTreeSearchMixin, MnkGameBotBase):
    def __init__(self, max_thinking_time, max_rollout, policy, exploration_const) -> None:
        super().__init__(max_thinking_time)
        self.max_rollout = max_rollout
        self.policy = policy
        self.c = exploration_const

    def inherit(self, last_moves: Tuple[Tuple[int, int], Tuple[int, int]]):
        # inherits previous tree root
        m1, m2 = last_moves
        if self.root is None:
            print("Initializing new tree...")
            return None
        try:
            print("Inheriting previous tree root...")
            new_root = self.root.children[m1].children[m2]
            return new_root
        except KeyError:
            print("Moves not found in previous tree. Initializing new tree...")
            return None

    def solve(self, board: MnkBoard, turn: int, start_time=None) -> Tuple[int, int]:
        start = start_time if start_time else time.time()
        self.root = MnkState(board, turn, self.policy, None, None)
        while time.time()-start < self.max_thinking_time and \
                self.total_rollout < self.max_rollout:
            self.loop()

    def get_results(self):
        best_child = None
        if self.total_rollout > 0 and len(self.root.children.values()) > 0:
            best_child = max(self.root.children.values(), key=self.score)
            children = []
            for child in self.root.children.values():
                children.append((child, self.score(child)))
            top_k = 5 if len(children) >= 5 else len(children)
            children.sort(key=lambda child: -child[1])
            print("\nTop %i moves:" % top_k)
            for child, score in children[:5]:
                print("Move:", child.last_move, "- score: %.4f - w: %i - n: %i" %
                    (score, child.r, child.n)
                )
        print("Played %i rollouts!" % self.rollout_count)
        print("Total: %i rollouts (inherited from previous trees)!" %
            self.total_rollout)
        return best_child.last_move if (best_child and self.total_rollout > 0) else (-1, -1)

    def selection(self):
        node = self.root
        while not node.is_leaf():
            selected_node = max(node.children.values(), key=lambda child:
                self.ucb(
                    child.r, child.n, self.c, node.n
                ))
            node = selected_node
        return node

    def choosing_policy(self, states):
        return random.choice(states)

    def expansion(self, node):
        if node.board.check_endgame() == 0:
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
        self.rollout_count += 1
        self.total_rollout += 1
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


def run(seed, tree: MonteCarloTreeSearchMnkGame, board: MnkBoard, turn: int,
        start_time: float):
    random.seed(seed)
    np.random.seed(seed)
    tree.solve(board, turn, start_time)
    return tree


def merge_trees(trees):
    for i in range(len(trees)-1):
        tree1 = trees[i]
        tree2 = trees[i+1]
        merge_nodes(tree1.root, tree2.root)
        tree2.rollout_count += tree1.rollout_count
        tree2.total_rollout += tree1.total_rollout


def merge_nodes(node1, node2, merge_children=True):
    for last_move, child in node2.children.items():
        if last_move in node1.children:
            # merges 2 nodes
            ret = child.merge(node1.children[last_move], node2)
            if not ret:
                raise Exception("Cannot merge 2 nodes:", child.last_move,
                    node1.children[last_move].last_move)
            # merges 2 nodes' offsprings
            if merge_children:
                merge_nodes(node1.children[last_move], child)
    for last_move, child in node1.children.items():
        if last_move not in node2.children:
            child.parent = node2
            node2.children[last_move] = child


def mcts_mnk_multi_proc(max_thinking_time, max_rollout, processes, policy,
        exploration_const, board, turn, last_moves):
    global last_tree
    start = time.time()
    args = []
    for i in range(processes):
        tree = MonteCarloTreeSearchMnkGame(max_thinking_time,
            max_rollout//processes, policy, exploration_const)
        if last_tree is not None and len(last_moves) == 2:
            root = last_tree.inherit(last_moves)
            if root is not None:
                tree.root = root
                tree.total_rollout = root.n
        random_seed = random.randint(0, 1<<32)
        args.append((random_seed, tree, board.duplicate(), turn, start))
    pool = multiprocessing.Pool(processes)
    trees = pool.starmap(run, args)
    trees = [tree for tree in trees if tree.total_rollout > 0]
    if not trees:
        return -1, -1
    print("\nSEPARATED TREES:")
    for tree in trees:
        tree.get_results()
    merge_trees(trees)
    final_tree = trees[-1]
    last_tree = final_tree
    print("\nCOMBINED RESULTS:")
    res = final_tree.get_results()
    last = time.time() - start
    print("Time: %.2f, games per second: %.2f" % (last, final_tree.rollout_count/last))
    return res


def mcts_mnk_single_process(max_thinking_time, max_rollout, policy,
        exploration_const, board, turn, last_moves):
    global last_tree
    start = time.time()
    tree = MonteCarloTreeSearchMnkGame(max_thinking_time, max_rollout,
            policy, exploration_const)
    if last_tree is not None and len(last_moves) == 2:
        root = last_tree.inherit(last_moves)
        if root is not None:
            tree.root = root
            tree.total_rollout = root.n
    tree.solve(board, turn)
    last_tree = tree
    res = tree.get_results()
    last = time.time() - start
    print("Time: %.2f, games per second: %.2f" % (last, tree.rollout_count/last))
    return res


def mcts_solve(max_thinking_time, max_rollout, processes, policy,
        exploration_const, board, turn, last_moves):
    if processes < 1:
        raise Exception("Invalid number of processes: {processes}!")
    elif processes > 1:
        return mcts_mnk_multi_proc(max_thinking_time, max_rollout, processes,
            policy, exploration_const, board, turn, last_moves)
    else:
        return mcts_mnk_single_process(max_thinking_time, max_rollout, policy,
            exploration_const, board, turn, last_moves)
