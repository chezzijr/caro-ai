from __future__ import annotations
from caro import Caro, Cell, Result
import numpy as np

# monte carlo tree search
class Node:
    def __init__(self, state: Caro, parent: Node | None = None):
        self.state = state
        self.parent = parent
        self.children: list[Node] = []
        self.visits = 0
        self.score = 0

    # because each child node is current node's next state
    def is_fully_expanded(self) -> bool:
        return len(self.children) == self.state.remaining_free_cells

class MonteCarloAgent:
    def __init__(self, game: Caro, cell: Cell, max_iter: int = 1000):
        self.game = game
        self.cell = cell
        self.max_iter = max_iter

    def mcts(self) -> tuple[int, int]:
        root = Node(self.game)
        for _ in range(self.max_iter):
            leaf = self.traverse(root)
            result = self.simulate(leaf)
            self.backpropagate(leaf, result)
        node = max(root.children, key=lambda x: x.visits)
        extract = np.argwhere(root.state.board != node.state.board)[0]
        return extract[0], extract[1]

    def generate_optimal_move(self) -> tuple[int, int]:
        return self.mcts()

    def traverse(self, node: Node) -> Node:
        while node.state.check_win() == Result.PENDING:
            if not node.is_fully_expanded():
                return self.expand(node)
            node = self.best_uct(node)
        return node

    def best_uct(self, node: Node) -> Node:
        return max(node.children, key=lambda x: x.score / x.visits + 2 * (np.log(node.visits) / x.visits) ** 0.5)

    def backpropagate(self, node: Node, result: int):
        while node:
            node.visits += 1
            node.score += result
            node = node.parent

    def simulate(self, node: Node) -> int:
        state = node.state.clone()
        while not state.check_win():
            state.move(*state.random_free_cell())

        if state.check_win() == Result.X_WIN and self.cell == Cell.X:
            return 1
        elif state.check_win() == Result.O_WIN and self.cell == Cell.O:
            return 1
        elif state.check_win() == Result.DRAW:
            return 0
        else:
            return -1

    def expand(self, node: Node) -> Node:
        state = node.state.clone()
        # get next state that is not existed in children
        mask = np.zeros((state.size, state.size), dtype=int)
        mask += state.board
        for child in node.children:
            mask += child.state.board
        row, col = np.argwhere(mask == Cell.EMPTY)[0]
        state.move(row, col)
        child = Node(state, node)
        node.children.append(child)
        return child
