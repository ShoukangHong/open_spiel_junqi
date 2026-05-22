"""Play Othello against a trained AlphaZero model via OpenSpiel."""

import os
import sys
import numpy as np
import pygame

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import utils
import pyspiel

# ── Model config (must match training run) ──────────────────────────────
CHECKPOINT_DIR = r"C:\Users\shouk\othello_train"
CHECKPOINT_STEP = 30
MODEL_TYPE = "resnet"
NN_WIDTH = 24
NN_DEPTH = 6
OBS_SHAPE = (3, 8, 8)
OUTPUT_SIZE = 65

# ── MCTS config ─────────────────────────────────────────────────────────
UCT_C = 1.41
MAX_SIMULATIONS = 100  # cranked up for actual play

# ── Pygame constants ────────────────────────────────────────────────────
ROWS = COLS = 8
SQ_SIZE = 80
WIDTH = HEIGHT = COLS * SQ_SIZE
PANEL_HEIGHT = 200
SCREEN_HEIGHT = HEIGHT + PANEL_HEIGHT

GREEN = (0, 128, 0)
DARK_GREEN = (0, 100, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)


def load_model():
    """Build AlphaZero model and load the checkpoint."""
    model = utils.api_selector("nnx").Model.build_model(
        model_type=MODEL_TYPE,
        input_shape=OBS_SHAPE,
        output_size=OUTPUT_SIZE,
        nn_width=NN_WIDTH,
        nn_depth=NN_DEPTH,
        weight_decay=1e-4,
        learning_rate=1e-3,
        path=CHECKPOINT_DIR,
    )
    model.load_checkpoint(CHECKPOINT_STEP)
    print(f"Loaded checkpoint-{CHECKPOINT_STEP} from {CHECKPOINT_DIR}")
    return model


def create_bot(game, model):
    """Create an MCTS bot backed by the AlphaZero model."""
    evaluator = az_evaluator.AlphaZeroEvaluator(game, model)
    bot = mcts.MCTSBot(
        game,
        UCT_C,
        MAX_SIMULATIONS,
        evaluator,
        solve=False,
        verbose=False,
        dont_return_chance_node=True,
    )
    return bot, evaluator


def obs_to_board(obs):
    """Convert [3, 8, 8] observation tensor to flat array: 1=black, -1=white, 0=empty."""
    obs = np.reshape(obs, (3, 8, 8))
    return (obs[1] - obs[2]).flatten().astype(int)


def get_piece_counts(board):
    black = int((board == 1).sum())
    white = int((board == -1).sum())
    return black, white


def draw_board(screen, board, legal_actions=None, hints=None):
    """Draw the 8x8 Othello board."""
    screen.fill(GREEN)

    # Grid
    for i in range(COLS + 1):
        pygame.draw.line(screen, BLACK, (i * SQ_SIZE, 0), (i * SQ_SIZE, HEIGHT), 2)
        pygame.draw.line(screen, BLACK, (0, i * SQ_SIZE), (WIDTH, i * SQ_SIZE), 2)

    # Pieces
    for row in range(ROWS):
        for col in range(COLS):
            idx = row * COLS + col
            cx = col * SQ_SIZE + SQ_SIZE // 2
            cy = row * SQ_SIZE + SQ_SIZE // 2

            if board[idx] == 1:
                pygame.draw.circle(screen, BLACK, (cx, cy), SQ_SIZE // 2 - 3)
            elif board[idx] == -1:
                pygame.draw.circle(screen, WHITE, (cx, cy), SQ_SIZE // 2 - 3)

    # Legal move indicators
    if legal_actions:
        for a in legal_actions:
            if a >= 64:  # pass
                continue
            row, col = a // COLS, a % COLS
            cx = col * SQ_SIZE + SQ_SIZE // 2
            cy = row * SQ_SIZE + SQ_SIZE // 2
            pygame.draw.circle(screen, DARK_GREEN, (cx, cy), 8)

    # Hint overlay (visit counts & win rate from MCTS)
    if hints:
        actions, visits, values = hints
        max_visit = max(visits) if visits else 1
        font = pygame.font.Font(None, 20)
        for a, vst, val in zip(actions, visits, values):
            if a >= 64:
                continue
            row, col = a // COLS, a % COLS
            cx = col * SQ_SIZE + SQ_SIZE // 2
            cy = row * SQ_SIZE + SQ_SIZE // 2
            ratio = vst / max_visit
            color = RED if ratio > 0.8 else (BLUE if ratio > 0.4 else GRAY)
            txt = font.render(str(vst), True, color)
            screen.blit(txt, (cx - txt.get_width() // 2, cy - txt.get_height() // 2))


def draw_panel(screen, black_count, white_count, current_player, message=""):
    """Draw info panel below the board."""
    font = pygame.font.Font(None, 30)
    y = HEIGHT + 10

    turn = "Black" if current_player == 0 else "White"
    texts = [
        f"Black: {black_count}    White: {white_count}",
        f"Turn: {turn}",
        message,
        f"H: hint   R: restart   Q: quit",
    ]
    for t in texts:
        surf = font.render(t, True, BLACK)
        screen.blit(surf, (10, y))
        y += 35


def choose_color(screen):
    """Menu: pick black or white."""
    font = pygame.font.Font(None, 40)
    btn_black = pygame.Rect(150, 250, 300, 60)
    btn_white = pygame.Rect(150, 350, 300, 60)

    while True:
        screen.fill(GREEN)
        title = font.render("Play as:", True, BLACK)
        screen.blit(title, (250, 150))

        pygame.draw.rect(screen, GRAY, btn_black)
        pygame.draw.rect(screen, GRAY, btn_white)
        screen.blit(font.render("Black (first)", True, WHITE), (210, 262))
        screen.blit(font.render("White (second)", True, WHITE), (210, 362))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_black.collidepoint(event.pos):
                    return 0
                if btn_white.collidepoint(event.pos):
                    return 1
        pygame.display.flip()


def game_over_screen(screen, board, black_count, white_count):
    """Show the result and wait for R (restart) or Q (quit)."""
    font = pygame.font.Font(None, 50)
    if black_count > white_count:
        msg = "Black wins!"
    elif white_count > black_count:
        msg = "White wins!"
    else:
        msg = "Draw!"

    while True:
        screen.fill(GREEN)
        draw_board(screen, board)

        # Overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        txt = font.render(msg, True, YELLOW)
        screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, HEIGHT // 2 - 40))

        small = pygame.font.Font(None, 30)
        r_txt = small.render("R - restart   Q - quit", True, WHITE)
        screen.blit(r_txt, (WIDTH // 2 - r_txt.get_width() // 2, HEIGHT // 2 + 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True   # restart
                if event.key == pygame.K_q:
                    return False  # quit
        pygame.display.flip()


def get_hints(bot, state):
    """Run MCTS and return (actions, visits, values) for each legal child."""
    root = bot.mcts_search(state)
    actions = []
    visits = []
    values = []
    for c in root.children:
        actions.append(c.action)
        visits.append(c.explore_count)
        values.append(c.total_reward / max(1, c.explore_count))
    return actions, visits, values


def main():
    print("Loading model...")
    model = load_model()

    game = pyspiel.load_game("othello")
    bot, evaluator = create_bot(game, model)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Othello vs AlphaZero")
    clock = pygame.time.Clock()

    while True:
        human_color = choose_color(screen)
        if human_color is None:
            break

        state = game.new_initial_state()
        showing_hints = False
        hints = None
        message = ""

        while not state.is_terminal():
            obs = state.observation_tensor(0)
            board = obs_to_board(obs)
            black_c, white_c = get_piece_counts(board)
            current = state.current_player()

            if current == human_color:
                # Human turn
                legal = state.legal_actions()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        if y < HEIGHT:
                            col, row = x // SQ_SIZE, y // SQ_SIZE
                            action = row * COLS + col
                            if action in legal:
                                state.apply_action(action)
                                bot.inform_action(state, current, action)
                                showing_hints = False
                                hints = None
                                message = ""
                                break
                            else:
                                message = "Illegal move"
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_h:
                            showing_hints = not showing_hints
                            if showing_hints:
                                hints = get_hints(bot, state)
                            else:
                                hints = None
                        elif event.key == pygame.K_r:
                            break  # restart
                        elif event.key == pygame.K_q:
                            pygame.quit()
                            sys.exit()

                if showing_hints and not state.is_terminal():
                    hints = get_hints(bot, state)
            else:
                # AI turn
                message = "AI thinking..."
                draw_board(screen, board, state.legal_actions(), hints)
                draw_panel(screen, black_c, white_c, current, message)
                pygame.display.flip()

                action = bot.step(state)
                state.apply_action(action)
                message = f"AI played: {action}"
                showing_hints = False
                hints = None

            # Render
            if not state.is_terminal():
                obs = state.observation_tensor(0)
                board = obs_to_board(obs)
                black_c, white_c = get_piece_counts(board)
                legal = state.legal_actions() if state.current_player() == human_color else None
                draw_board(screen, board, legal, hints)
                draw_panel(screen, black_c, white_c, state.current_player(), message)

            clock.tick(30)
            pygame.display.flip()

        # Game over
        obs = state.observation_tensor(0)
        board = obs_to_board(obs)
        black_c, white_c = get_piece_counts(board)
        draw_board(screen, board)
        pygame.display.flip()

        if not game_over_screen(screen, board, black_c, white_c):
            break

    pygame.quit()


if __name__ == "__main__":
    main()
