import time
import cv2
import numpy as np
from PIL import Image
from src.tetris import Tetris


LEFT_KEYS = {ord("a"), ord("A"), 2424832}
RIGHT_KEYS = {ord("d"), ord("D"), 2555904}
ROTATE_KEYS = {ord("w"), ord("W"), 2490368}
SOFT_DROP_KEYS = {ord("s"), ord("S"), 2621440}
HARD_DROP_KEYS = {32}  # spacja
QUIT_KEYS = {ord("q"), ord("Q"), 27}  # q lub esc


def collides(env, piece, pos):
    for y in range(len(piece)):
        for x in range(len(piece[y])):
            if piece[y][x] == 0:
                continue

            board_x = pos["x"] + x
            board_y = pos["y"] + y

            if board_x < 0 or board_x >= env.width:
                return True
            if board_y < 0 or board_y >= env.height:
                return True
            if env.board[board_y][board_x] != 0:
                return True
    return False


def try_move(env, dx, dy):
    new_pos = {"x": env.current_pos["x"] + dx, "y": env.current_pos["y"] + dy}
    if not collides(env, env.piece, new_pos):
        env.current_pos = new_pos
        return True
    return False


def try_rotate(env):
    rotated = env.rotate(env.piece)
    if not collides(env, rotated, env.current_pos):
        env.piece = rotated
        return True
    return False


def lock_piece(env):
    if env.truncate(env.piece, env.current_pos):
        env.gameover = True

    env.board = env.store(env.piece, env.current_pos)

    lines_cleared, env.board = env.check_cleared_rows(env.board)
    score = 1 + (lines_cleared ** 2) * env.width
    env.score += score
    env.tetrominoes += 1
    env.cleared_lines += lines_cleared

    if not env.gameover:
        env.new_piece()
    else:
        env.score -= 2


def hard_drop(env):
    while try_move(env, 0, 1):
        pass
    lock_piece(env)


def draw_piece_preview(canvas, env, top_left_x, top_left_y):
    preview_block = 22
    preview = env.piece
    for y in range(len(preview)):
        for x in range(len(preview[y])):
            val = preview[y][x]
            if val == 0:
                continue

            color_rgb = env.piece_colors[val]
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            x1 = top_left_x + x * preview_block
            y1 = top_left_y + y * preview_block
            x2 = x1 + preview_block - 2
            y2 = y1 + preview_block - 2

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 40, 40), 1)


def render_window(env, drop_interval_ms):
    board = env.get_current_board_state() if not env.gameover else env.board
    img = [env.piece_colors[p] for row in board for p in row]
    img = np.array(img).reshape((env.height, env.width, 3)).astype(np.uint8)
    img = img[..., ::-1]
    img = Image.fromarray(img, "RGB")
    img = np.array(
        img.resize((env.width * env.block_size, env.height * env.block_size), 0)
    )

    img[[i * env.block_size for i in range(env.height)], :, :] = 0
    img[:, [i * env.block_size for i in range(env.width)], :] = 0

    side_width = 320
    side = np.ones((env.height * env.block_size, side_width, 3), dtype=np.uint8) * np.array(
        [245, 245, 252], dtype=np.uint8
    )

    canvas = np.concatenate((img, side), axis=1)

    # panel background accents
    panel_x = env.width * env.block_size
    cv2.rectangle(canvas, (panel_x, 0), (panel_x + side_width, env.height * env.block_size), (245, 245, 252), -1)
    cv2.rectangle(canvas, (panel_x + 10, 10), (panel_x + side_width - 10, 135), (230, 230, 245), -1)
    cv2.rectangle(canvas, (panel_x + 10, 150), (panel_x + side_width - 10, 355), (232, 238, 248), -1)

    x0 = panel_x + 22
    y = 38
    color_dark = (70, 45, 120)
    color_text = (80, 60, 120)

    def put(text, scale=0.68, color=color_text, thick=1, dy=28):
        nonlocal y
        cv2.putText(
            canvas,
            text,
            (x0, y),
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            color,
            thick,
            cv2.LINE_AA,
        )
        y += dy

    put("HUMAN TETRIS", scale=0.9, color=color_dark, thick=2, dy=34)
    put(f"Score: {env.score}", scale=0.72, dy=30)
    put(f"Pieces: {env.tetrominoes}", scale=0.72, dy=30)
    put(f"Lines: {env.cleared_lines}", scale=0.72, dy=30)
    put(f"Speed: {drop_interval_ms} ms", scale=0.68, dy=34)

    # next/current piece preview
    cv2.putText(
        canvas,
        "Current piece",
        (x0, 182),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        color_dark,
        2,
        cv2.LINE_AA,
    )
    draw_piece_preview(canvas, env, x0, 198)

    # controls
    controls_y = 380
    cv2.putText(
        canvas,
        "Controls",
        (x0, controls_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        color_dark,
        2,
        cv2.LINE_AA,
    )

    controls = [
        "A / Left   - move left",
        "D / Right  - move right",
        "W / Up     - rotate",
        "S / Down   - soft drop",
        "Space      - hard drop",
        "Q / Esc    - quit",
    ]

    yy = controls_y + 32
    for line in controls:
        cv2.putText(
            canvas,
            line,
            (x0, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color_text,
            1,
            cv2.LINE_AA,
        )
        yy += 28

    if env.gameover:
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (25, env.height * env.block_size // 2 - 50),
            (env.width * env.block_size - 25, env.height * env.block_size // 2 + 50),
            (25, 25, 25),
            -1,
        )
        canvas = cv2.addWeighted(overlay, 0.60, canvas, 0.40, 0)
        cv2.putText(
            canvas,
            "GAME OVER",
            (50, env.height * env.block_size // 2 + 12),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Human Tetris", canvas)


def main():
    env = Tetris()
    env.reset()

    cv2.namedWindow("Human Tetris", cv2.WINDOW_AUTOSIZE)

    drop_interval_ms = 400
    last_drop = time.time()

    while True:
        render_window(env, drop_interval_ms)

        key = cv2.waitKeyEx(30)

        if key in QUIT_KEYS:
            break

        if not env.gameover:
            if key in LEFT_KEYS:
                try_move(env, -1, 0)
            elif key in RIGHT_KEYS:
                try_move(env, 1, 0)
            elif key in ROTATE_KEYS:
                try_rotate(env)
            elif key in SOFT_DROP_KEYS:
                moved = try_move(env, 0, 1)
                if not moved:
                    lock_piece(env)
                last_drop = time.time()
            elif key in HARD_DROP_KEYS:
                hard_drop(env)
                last_drop = time.time()

            now = time.time()
            if (now - last_drop) * 1000 >= drop_interval_ms:
                moved = try_move(env, 0, 1)
                if not moved:
                    lock_piece(env)
                last_drop = now

        else:
            if key != -1:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()