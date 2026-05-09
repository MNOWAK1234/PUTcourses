"""
@original author: Viet Nguyen <nhviet1009@gmail.com>
@refactoring author: Michael Greif <Greifenhard>
"""

import cv2
import torch
import random
import numpy as np
from PIL import Image
from typing import Tuple, Dict
from matplotlib import style

style.use("ggplot")

class Tetris:
    piece_colors = [
        (0, 0, 0),          # Empty space
        (255, 255, 0),      # O piece, yellow
        (147, 88, 254),     # T piece, purple
        (54, 175, 144),     # S piece, green
        (255, 0, 0),        # Z piece, red
        (102, 217, 238),    # I piece, cyan
        (254, 151, 32),     # L piece, orange
        (0, 0, 255),        # J piece, blue
    ]
    pieces = [
        [[1, 1], [1, 1]],       # O piece
        [[0, 2, 0], [2, 2, 2]], # T piece
        [[0, 3, 3], [3, 3, 0]], # S piece
        [[4, 4, 0], [0, 4, 4]], # Z piece
        [[5, 5, 5, 5]],         # I piece
        [[0, 0, 6], [6, 6, 6]], # L piece
        [[7, 0, 0], [7, 7, 7]], # J piece
    ]

    def __init__(self, height:int=20, width:int=10, block_size:int=30) -> None:
        """Initialize the Tetris game with specified height, width, and block size.
        Args:
            height (int, optional): Height of the Tetris board. Defaults to 20.
            width (int, optional): Width of the Tetris board. Defaults to 10.
            block_size (int, optional): Blocksize for tetrominoes. Defaults to 30.
        """
        self.height = height
        self.width = width
        self.block_size = block_size
        self.extra_board = np.ones(
            (self.height * self.block_size, self.width * int(self.block_size / 2), 3),
            dtype=np.uint8,
        ) * np.array([204, 204, 255], dtype=np.uint8)
        self.text_color = (200, 20, 220)
        self.reset()

    def reset(self) -> torch.FloatTensor:
        """Reset the Tetris game to its initial state.
        Returns:
            torch.FloatTensor: A tensor containing the initial state properties of the Tetris board.
        """
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False
        return self.get_state_properties(self.board)

    def get_state_properties(self, board:list[list[int]]) -> torch.FloatTensor:
        """Calculate and return the state properties of the Tetris board.
        Args:
            board (list): The current state of the Tetris board.
        Returns:
            torch.FloatTensor: A tensor containing the properties of the board state.
        """
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def check_cleared_rows(self, board:list[list[int]]) -> Tuple[int, list]:
        """Check for cleared rows in the Tetris board and remove them.
        Args:
            board (list): The current state of the Tetris board.
        Returns:
            Tuple: A tuple containing the number of cleared rows and the updated board.
        """
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board
    
    def get_holes(self, board:list[list[int]]) -> int:
        """ Calculate number of holes in the Tetris board.
        Args:
            board (list): The current state of the Tetris board.
        Returns:
            int: The number of holes in the board.
        """
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0: row += 1
            num_holes += len([x for x in col[row + 1 :] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board:list[list[int]]) -> Tuple[int, int]:
        """Calculate the bumpiness and height of the Tetris board.
        Args:
            board (list): The current state of the Tetris board.
        Returns:
            Tuple ([int, int]): A tuple containing the total bumpiness and total height of the board.
        """
        board = np.array(board)
        mask = (board != 0)
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        total_bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))
        return total_bumpiness, total_height

    def get_next_states(self) -> Dict[Tuple[int, int], torch.FloatTensor]:
        """Generate all possible next states for the current piece in the Tetris game.
        Returns:
            dict ({Tuple[int,int], torch.FloatTensor}): A dictionary where keys are tuples of (x, rotation) and values are the state properties of the board.
        """
        states = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]
        
        # Determine the number of rotations based on the piece type
        if piece_id == 0: num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4: num_rotations = 2
        else: num_rotations = 4

        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece, pos = [row[:] for row in curr_piece], {"x": x, "y": 0}
                while not self.check_collision(piece, pos): pos["y"] += 1
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_state_properties(board)
            curr_piece = self.rotate(curr_piece)
        return states

    def get_current_board_state(self) -> list:
        """Get the current state of the Tetris board with the active piece placed."""
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                if self.piece[y][x] != 0:
                    board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        return board

    def new_piece(self) -> None:
        """Generate a new piece for the Tetris game. If the bag is empty, refill it with all pieces."""
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece:list[list[int]], pos:dict[str,int]) -> bool:
        """Check if the current piece collides with the board or goes out of bounds.
        Args:
            piece (list[list[int]]): The current piece represented as a 2D list.
            pos (dict[str, int]): The position of the piece on the board with keys 'x' and 'y'.
        Returns:
            bool: True if there is a collision, False otherwise.
        """
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if (future_y + y > self.height - 1
                    or self.board[future_y + y][pos["x"] + x]
                    and piece[y][x]):
                    return True
        return False

    def truncate(self, piece:list[list[int]], pos:dict[str,int]) -> bool:
        """Truncate the piece if it collides with the top of the board.
        Args:
            piece (list[list[int]]): The current piece represented as a 2D list.
            pos (dict[str, int]): The position of the piece on the board with keys 'x' and 'y'.
        Returns:
            bool: True if the piece is truncated at the top, False otherwise.
        """
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if (self.board[pos["y"] + y][pos["x"] + x]
                            and piece[y][x]
                            and y > last_collision_row):
                            last_collision_row = y
        return gameover

    def store(self, piece:list[list[int]], pos:dict[str,int]) -> list[list[int]]:
        """Store the current piece on the Tetris board at the specified position.
        Args:
            piece (list[list[int]]): The current piece represented as a 2D list.
            pos (dict[str, int]): The position of the piece on the board with keys 'x' and 'y'.
        Returns:
            list[list[int]]: The updated Tetris board with the piece stored.
        """
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def remove_row(self, board:list[list[int]], indices:list) -> list[list[int]]:
        """Remove specified rows from the Tetris board and shift the remaining rows down.
        Args:
            board (list): The current state of the Tetris board.
            indices (list): A list of indices of rows to be removed.
        Returns:
            list: The updated Tetris board with specified rows removed and remaining rows shifted down.
        """
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

    def rotate(self, piece:list[list[int]]) -> list[list[int]]:
        """Rotate the current piece 90 degrees clockwise.
        Args:
            piece (list[list[int]]): The current piece represented as a 2D list.
        Returns:
            list[list[int]]: The rotated piece represented as a 2D list.
        """
        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array
    
    def step(self, action:tuple[int,int], render:bool=True, video:bool=False):
        """Perform a step in the Tetris game with the given action.
        
        Args:
            action (tuple): A tuple containing the x-coordinate and number of rotations.
            render (bool, optional): If True, renders the game state. Defaults to True.
            video (bool, optional): If True, saves the rendered frame to a video file. Defaults to False.
        
        Returns:
            tuple: A tuple containing the reward for this move and a boolean indicating if the game is over.
        """
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            if render:
                self.render(video)

        if self.truncate(self.piece, self.current_pos):
            self.gameover = True  # Game over if piece is truncated at the top

        self.board = self.store(self.piece, self.current_pos)

        lines_cleared, self.board = self.check_cleared_rows(self.board)

        # Retro classic scoring (NES/Game Boy style, level = 1)
        line_scores = {
            0: 0,
            1: 40,
            2: 100,
            3: 300,
            4: 1200,
        }
        score = line_scores.get(lines_cleared, 0)

        self.score += score
        self.tetrominoes += 1
        self.cleared_lines += lines_cleared
        
        if not self.gameover:
            self.new_piece()
        else:
            self.score -= 2

        return score, self.gameover

    def render(self, video:bool=False):
        """Render the current state of the Tetris game.

        Args:
            video (bool, optional): If True, saves the rendered frame to a video file. Defaults to False.
        """
        if not self.gameover: img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        else: img = [self.piece_colors[p] for row in self.board for p in row]
        
        img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        img = np.array(img.resize((self.width * self.block_size, self.height * self.block_size), 0))
        img[[i * self.block_size for i in range(self.height)], :, :] = 0
        img[:, [i * self.block_size for i in range(self.width)], :] = 0

        img = np.concatenate((img, self.extra_board), axis=1)

        def putText(img, text, org):
            cv2.putText(img, text, org, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        
        putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size))
        putText(img, str(self.score), (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size))
        
        putText(img, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size))
        putText(img, str(self.tetrominoes), (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size))
        
        putText(img, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 7 * self.block_size))
        putText(img, str(self.cleared_lines), (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size))
        
        if video: video.write(img)

        cv2.imshow("RL Tetris", img)
        cv2.waitKey(1)
