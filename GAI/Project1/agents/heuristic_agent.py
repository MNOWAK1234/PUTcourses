import random
import numpy as np


class HeuristicAgent:
    def __init__(self, state_size=None, seed=123):
        self.name = "heuristic"
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def _column_heights(self, board):
        heights = []
        h = len(board)
        w = len(board[0])

        for x in range(w):
            col_height = 0
            for y in range(h):
                if board[y][x] != 0:
                    col_height = h - y
                    break
            heights.append(col_height)
        return heights

    def _count_holes(self, board):
        holes = 0
        h = len(board)
        w = len(board[0])

        for x in range(w):
            seen_block = False
            for y in range(h):
                if board[y][x] != 0:
                    seen_block = True
                elif seen_block:
                    holes += 1
        return holes

    def _bumpiness_and_total_height(self, heights):
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness, sum(heights)

    def _left_well_depth(self, heights):
        if len(heights) < 2:
            return 0
        return max(0, heights[1] - heights[0])

    def _left_well_blocked(self, board):
        """
        Kara, jeśli lewa kolumna ma dużo bloków wysoko,
        czyli studnia nie jest już sensownie otwarta.
        """
        h = len(board)
        blocked = 0
        for y in range(h):
            if board[y][0] != 0:
                blocked += 1
        return blocked

    def _count_lines_cleared(self, old_board, new_board, env):
        old_full = sum(1 for row in old_board if 0 not in row)
        new_full = sum(1 for row in new_board if 0 not in row)
        # realnie i tak po check_cleared_rows dostaniemy liczbę wyczyszczonych linii osobno,
        # więc tej funkcji nie używamy; zostawiona pomocniczo
        return max(0, old_full - new_full)

    def _simulate_action(self, env, action):
        x, num_rotations = action

        piece = [row[:] for row in env.piece]
        for _ in range(num_rotations):
            piece = env.rotate(piece)

        pos = {"x": x, "y": 0}

        while not env.check_collision(piece, pos):
            pos["y"] += 1

        env.truncate(piece, pos)
        board_after_store = env.store(piece, pos)
        lines_cleared, board_after_clear = env.check_cleared_rows(board_after_store)

        return board_after_clear, lines_cleared

    def _evaluate_board(self, board, lines_cleared):
        heights = self._column_heights(board)
        holes = self._count_holes(board)
        bumpiness, total_height = self._bumpiness_and_total_height(heights)
        max_height = max(heights) if heights else 0

        left_well_depth = self._left_well_depth(heights)
        left_well_blocked = self._left_well_blocked(board)

        score = 0.0
        score += 120.0 * lines_cleared
        score -= 45.0 * holes
        score -= 3.0 * bumpiness
        score -= 2.0 * total_height
        score -= 8.0 * max_height
        score += 12.0 * left_well_depth
        score -= 20.0 * left_well_blocked

        return score

    def select_action(self, current_board_features, env, epsilon_override=0.0):
        possible_actions = list(env.get_next_states().keys())

        best_score = -float("inf")
        best_actions = []

        for action in possible_actions:
            board_after, lines_cleared = self._simulate_action(env, action)
            score = self._evaluate_board(board_after, lines_cleared)

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        chosen_action = self.rng.choice(best_actions)
        return chosen_action, best_score