import math
from abc import ABC, abstractmethod
from utils.mixin import PerfMonitorMixin


class MonteCarloTreeSearchMixin(ABC, PerfMonitorMixin):
    
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
        return (w / n + c * math.sqrt(math.log(t)/n)) if n != 0 else math.inf
