import time
import random
import multiprocessing
import numpy as np
from queue import Empty
from typing import Tuple
from mcts.mcts import MonteCarloTreeSearchMixin
from .mnk_bot_base import MnkGameBotBase
from .board import MnkBoard


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
        for k in range(len(pos)):
            i, j = pos[k]
            new_board = self.board.duplicate()
            new_board.board[j][i] = 3-self.turn
            new_state = MnkState(new_board, 3-self.turn, self.policy, 
                (i, j), self)
            states.append(new_state)
        return states

    def merge(self, other, parent, in_place=True):
        # self and other must have similar parents to merge
        if self.last_move != other.last_move:
            return False
        if in_place:
            state = self
            state.n += other.n
            state.r += other.r
        else:
            state = MnkState(self.board.duplicate(), self.turn, self.policy,
                self.last_move, parent, list(), self.n+other.n, self.r+other.r)
        return state
        

    @staticmethod
    def simple_rollout_policy(board):
        # completely random moves
        pos = board.get_possible_pos()
        if len(pos) == 0:
            return -1, -1
        index = random.randrange(0, len(pos))
        return pos[index]

    def prob_rollout_policy(self):
        # cells near previously marked cell get greater probability to be chosen
        pass

    def rollout(self):
        test_board = self.board.duplicate()
        turn = self.turn
        res = test_board.check_endgame()
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
            res = test_board.check_endgame()
        return res


class MonteCarloTreeSearchMnkGame(MonteCarloTreeSearchMixin, MnkGameBotBase):
    def __init__(self, max_thinking_time, max_rollout, policy, exploration_const) -> None:
        super().__init__(max_thinking_time)
        self.max_rollout = max_rollout
        self.policy = policy
        self.c = exploration_const
        self.root = None
        self.rollout_count = 0

    def solve(self, board: MnkBoard, turn: int, start_time=None) -> Tuple[int, int]:
        start = start_time if start_time else time.time()
        self.root = MnkState(board, turn, self.policy, None, None)
        self.rollout_count = 0
        while time.time()-start < self.max_thinking_time and \
                self.rollout_count < self.max_rollout:
            self.loop()

    def get_results(self):
        if self.rollout_count > 0:
            best_child = max(self.root.children.values(), key=self.score)
            children = list()
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
        return best_child.last_move if self.rollout_count > 0 else (-1, -1)

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
            old_states = list()
            for state in next_states:
                if state.last_move not in node.children:
                    node.children[state.last_move] = state
                old_states.append(node.children[state.last_move])
            if old_states:
                node = self.choosing_policy(old_states)
        return node

    def simulation(self, node):
        self.rollout_count += 1
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


def merge_nodes(node1, node2):
    for last_move, child in node2.children.items():
        if last_move in node1.children:
            # merges 2 nodes' offsprings
            merge_nodes(node1.children[last_move], child)
            # merges 2 nodes
            child.merge(node1.children[last_move], node2)
    for last_move, child in node1.children.items():
        if last_move not in node2.children:
            child.parent = node2
            node2.children[last_move] = child


def mcts_mnk_multi_proc(max_thinking_time, max_rollout, processes, policy, 
        exploration_const, board, turn):
    start = time.time()
    args = list()
    for i in range(processes):
        tree = MonteCarloTreeSearchMnkGame(max_thinking_time, 
            max_rollout//processes, policy, exploration_const)
        args.append((i, tree, board.duplicate(), turn, start))
    pool = multiprocessing.Pool(processes)
    trees = pool.starmap(run, args)
    trees = [tree for tree in trees if tree.rollout_count > 0]
    if not trees:
        return -1, -1
    print("\nSeparated trees:")
    for tree in trees:
        tree.get_results()
    merge_trees(trees)
    final_tree = trees[-1]
    print("\nCombined results:")
    res = final_tree.get_results()
    last = time.time() - start
    print("Time: %.2f, games per second: %.2f" % 
        (last, final_tree.rollout_count/last))
    return res


def mcts_mnk_single_process(max_thinking_time, max_rollout, policy, exploration_const,
        board, turn):
    start = time.time()
    tree = MonteCarloTreeSearchMnkGame(max_thinking_time, max_rollout, policy,
            exploration_const)
    tree.solve(board, turn)
    res = tree.get_results()
    last = time.time() - start
    print("Time: %.2f, games per second: %.2f" % 
        (last, tree.rollout_count/last))
    return res

def mcts_solve(max_thinking_time, max_rollout, processes, policy, 
        exploration_const, board, turn):
    if processes < 1:
        raise Exception("Invalid number of processes: {processes}!")
    elif processes > 1:
        return mcts_mnk_multi_proc(max_thinking_time, max_rollout, processes, 
            policy, exploration_const, board, turn)
    else:
        return mcts_mnk_single_process(max_thinking_time, max_rollout, policy, 
            exploration_const, board, turn)
