import math
import cython
from abc import ABC, abstractmethod
from utils.perf_monitor import PerfMonitorMixin


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
        ret = (w / n + c * math.sqrt(math.log(t)/n))
        return ret

    @staticmethod
    def score(node):
        ret = node.r/node.n
        return ret

    @staticmethod
    def ucb(node, c):
        return node.r/node.n + c * math.sqrt(math.log(node.parent.n)/node.n)
