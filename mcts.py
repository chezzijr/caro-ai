from __future__ import annotations
from caro import Caro, Cell

# monte carlo tree search
class Node:
    def __init__(self, state: Caro, parent: Node | None = None):
        self.state = state
        self.parent = parent
        self.children: list[Node] = []
        self.visits = 0
        self.score = 0

    def is_fully_expanded(self) -> bool:
        return len(self.children) == self.state.remaining_free_cells

class MonteCarloAgent:
    def __init__(self, game: Caro, cell: Cell, max_iter: int = 1000):
        self.game = game
        self.cell = cell
        self.max_iter = max_iter

