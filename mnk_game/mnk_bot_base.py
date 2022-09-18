from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod
from .board import MnkBoard


class MnkGameBotBase(ABC):
    def __init__(self, thinking_time, processes) -> None:
        super().__init__()
        self.thinking_time = thinking_time
        self.processes = processes

    @abstractmethod
    def solve(self, board: MnkBoard, turn: int) -> Tuple[int, int]:
        pass
