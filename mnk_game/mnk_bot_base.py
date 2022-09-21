from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod
from .board import MnkBoard


class MnkGameBotBase(ABC):
    def __init__(self, max_thinking_time) -> None:
        super().__init__()
        self.max_thinking_time = max_thinking_time

    @abstractmethod
    def solve(self, board: MnkBoard, turn: int) -> Tuple[int, int]:
        pass
