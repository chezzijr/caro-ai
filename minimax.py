from caro import Caro, Cell, Result
import numpy as np

INF = int(1e9)

# self is maximizer, opponent is minimizer
class MinimaxAgent:
    def __init__(self, game: Caro, cell: Cell, depth: int = 3):
        self.game = game
        self.cell = cell
        self.depth = depth

    # count how many consecutive cells
    # this is neutral to the player
    # meaning that it will return the same value for both player
    def evaluate_at(self, row, col, visited: np.ndarray) -> int:
        if visited[row, col]:
            return 0
        visited[row, col] = 1
        score = 0

        # row
        r1, r2 = self.game.get_row_consecutive(row, col)
        is_blocked_left = r1 == 0 or self.game.board[row, r1 - 1] != Cell.EMPTY
        is_blocked_right = r2 == self.game.size - 1 or self.game.board[row, r2 + 1] != Cell.EMPTY
        # print("Row consecutive cells:", r2 - r1 + 1)
        score += (r2 - r1 + 1) ** 2 * (2 - is_blocked_left - is_blocked_right)
        visited[row, r1:r2 + 1] = 1

        # col
        c1, c2 = self.game.get_col_consecutive(row, col)
        is_blocked_up = c1 == 0 or self.game.board[c1 - 1, col] != Cell.EMPTY
        is_blocked_down = c2 == self.game.size - 1 or self.game.board[c2 + 1, col] != Cell.EMPTY
        # print("Column consecutive cells:", c2 - c1 + 1)
        score += (c2 - c1 + 1) ** 2 * (2 - is_blocked_up - is_blocked_down)
        visited[c1:c2 + 1, col] = 1

        # main diag
        d1, d2 = self.game.get_main_diag_consecutive(row, col)
        r1, c1 = row - d1, col - d1
        r2, c2 = row + d2, col + d2
        is_blocked_up_left = (r1 == 0 or c1 == 0) or self.game.board[r1 - 1, c1 - 1] != Cell.EMPTY
        is_blocked_down_right = (r2 == self.game.size - 1 or c2 == self.game.size - 1) or self.game.board[r2 + 1, c2 + 1] != Cell.EMPTY
        # print("Main diagonal consecutive cells:", d2 - d1 + 1)
        score += (d2 + d1 + 1) ** 2 * (2 - is_blocked_up_left - is_blocked_down_right)
        for i in range(-d1, d2 + 1):
            visited[row + i, col + i] = 1

        # anti diag
        ad1, ad2 = self.game.get_anti_diag_consecutive(row, col)
        r1, c1 = row - ad1, col + ad1
        r2, c2 = row + ad2, col - ad2
        is_blocked_up_right = (r1 == 0 or c1 == self.game.size - 1) or self.game.board[r1 - 1, c1 + 1] != Cell.EMPTY
        is_blocked_down_left = (r2 == self.game.size - 1 or c2 == 0) or self.game.board[r2 + 1, c2 - 1] != Cell.EMPTY
        # print("Anti diagonal consecutive celss:", ad2 - ad1 + 1)
        score += (ad2 + ad1 + 1) ** 2 * (2 - is_blocked_up_right - is_blocked_down_left)
        for i in range(-ad1, ad2 + 1):
            visited[row + i, col - i] = 1

        # print("Score:", score)
        return score

    # this is non-neutral
    # so it will return positive value if the current agent
    # and negative value if the opponent
    def evaluate(self) -> int:
        visited = np.zeros((self.game.size, self.game.size), dtype=np.uint8)
        score = 0
        for row in range(self.game.size):
            for col in range(self.game.size):
                if visited[row, col]:
                    continue
                cell = self.game.board[row, col]
                if cell == Cell.EMPTY:
                    continue
                if cell == self.cell:
                    score += self.evaluate_at(row, col, visited)
                else:
                    score -= self.evaluate_at(row, col, visited)

        return score

    def minimax(self, depth: int, alpha: int, beta: int, is_maximizer: bool) -> tuple[int, tuple[int, int] | None]:
        result = self.game.check_win()
        if result != Result.PENDING:
            if self.cell == Cell.X and result == Result.X_WIN:
                return INF, None
            elif self.cell == Cell.O and result == Result.O_WIN:
                return INF, None
            elif result == Result.DRAW:
                return -INF // 2, None
            else:
                return -INF, None

        if depth == 0:
            return self.evaluate(), None

        # # there is no breaking outside loop in python
        # # so we use this flag to break the loop
        # break_flag = False
        surroundings = self.game.get_surroundings(1)
        best_move = None
        if is_maximizer:
            best = -INF
            for row, col in surroundings:
                self.game.move(row, col)

                if best < (val := self.minimax(depth - 1, alpha, beta, not is_maximizer)[0]):
                    best = val
                    best_move = (row, col)

                self.game.unmove(row, col)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break

            return best, best_move
        else:
            best = INF
            for row, col in surroundings:
                self.game.move(row, col)

                if best > (val := self.minimax(depth - 1, alpha, beta, not is_maximizer)[0]):
                    best = val
                    best_move = (row, col)

                self.game.unmove(row, col)
                beta = min(beta, best)
                if beta <= alpha:
                    break

            return best, best_move

    def generate_random_move(self) -> tuple[int, int]:
        empty_cells = np.argwhere(self.game.board == Cell.EMPTY)
        return empty_cells[np.random.randint(len(empty_cells))]

    def generate_optimal_move(self) -> tuple[int, int]:
        _, move = self.minimax(self.depth, -INF, INF, True)
        if move is None:
            return self.generate_random_move()
        return move
