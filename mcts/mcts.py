import math
import cython
from abc import ABC, abstractmethod
from utils.mixin import PerfMonitorMixin
from .mcts_algorithms import ucb, score


class MonteCarloTreeSearchMixin(ABC, PerfMonitorMixin):

    root = None
    rollout_count = 0
    total_rollout = 0
    
    def loop(self) -> None:
        node = self.selection()
        node = self.expansion(node)
        winner = self.simulation(node)
        self.backpropagation(node, winner)

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def expansion(self):
        pass

    @abstractmethod
    def simulation(self):
        pass

    @abstractmethod
    def backpropagation(self):
        pass

    @staticmethod
    def ucb(w, n, c, t):
        ret = ucb(w, n, c, t)
        # ret = (w / n + c * math.sqrt(math.log(t)/n)) if n != 0 else math.inf
        return ret

    @staticmethod
    def score(node):
        ret = score(node.r, node.n)
        # ret = node.r/node.n if node.n != 0 else -math.inf
        return ret
