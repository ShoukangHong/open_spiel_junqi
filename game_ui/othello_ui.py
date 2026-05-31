"""Play Othello against a trained PyTorch AlphaZero model via OpenSpiel."""

import json
import os
import sys

import numpy as np
import pygame

# Make train/ imports work from this directory
_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

import pyspiel
from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.evaluator import PyTorchEvaluator
from train.batch_mcts.mcts import BatchMCTS
from train.model.othello_resnet import Model, OthelloResNet

# ── Config — paths only, model settings read from checkpoint dir ────────
# CHECKPOINT_DIR = r"C:\Users\shouk\othello_train_v2"
# CHECKPOINT_DIR = r"C:\Users\shouk\othello_train_cloud"
CHECKPOINT_DIR = r"C:\Users\shouk\othello_train\cloud_wdl_w"
CHECKPOINT_STEP = 250             # checkpoint step to load (must exist)
MCTS_SIMULATIONS = 256          # MCTS search budget per move
HINT_MAX_SIM = 12800
MCTS_BATCH_SIZE = 8             # leaf evaluation batch size
UCT_C = 1.41


def load_model(game):
    """Build PyTorch OthelloResNet using core model builder."""
    config_path = os.path.join(CHECKPOINT_DIR, "train_config.json")
    with open(config_path) as f:
        train_cfg = json.load(f)
    from train.core.model_builder import build_othello_model
    train_cfg["path"] = CHECKPOINT_DIR  # override saved config
    model = build_othello_model(game, train_cfg)
    model.load_checkpoint(CHECKPOINT_STEP)
    print(f"Loaded checkpoint-{CHECKPOINT_STEP} from {CHECKPOINT_DIR}")
    print(f"  params={model.num_trainable_variables}")
    return model


def create_bot(game, model):
    """Create a BatchMCTS bot backed by the PyTorch model."""
    evaluator = PyTorchEvaluator(game, model)
    mcts_cfg = MCTSConfig(
        max_simulations=MCTS_SIMULATIONS,
        batch_size=MCTS_BATCH_SIZE,
        uct_c=UCT_C,
        policy_epsilon=0,
        verbose=False,
    )
    bot = BatchMCTS(game, mcts_cfg, evaluator,
                    random_state=np.random.RandomState())
    return bot, evaluator


# ── Pygame constants ────────────────────────────────────────────────────
ROWS = COLS = 8
SQ_SIZE = 80
WIDTH = HEIGHT = COLS * SQ_SIZE
PANEL_HEIGHT = 120
SCREEN_HEIGHT = HEIGHT + PANEL_HEIGHT

GREEN = (0, 128, 0)
DARK_GREEN = (0, 100, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)


def obs_to_board(obs):
    """Convert [4, 8, 8] observation tensor to flat array: 1=black, -1=white, 0=empty."""
    obs = np.reshape(obs, (4, 8, 8))
    return (obs[1] - obs[2]).flatten().astype(int)

def print_state(state):
  """Print 4-channel observation tensor: 4 grids side by side."""
  print(state)
  # obs = np.reshape(state.observation_tensor(0), (4, 8, 8))
  # for row in range(8):
  #     parts = []
  #     for ch in range(4):
  #         parts.append(" ".join(
  #             "." if obs[ch, row, col] == 0 else str(int(obs[ch, row, col]))
  #             for col in range(8)))
  #     print("  |  ".join(parts))
  # print()

def print_mcts_info(root, state, evaluator=None):
    """Print MCTS root policy, raw NN prior, and value."""
    player = state.current_player()
    player_name = "BLACK(p0)" if player == 0 else "WHITE(p1)"
    print(f"  ── MCTS  {player_name} ──")

    # Value — proven outcome if solved
    if root.outcome is not None:
        root_val = root.outcome[player]
    else:
        root_val = root.total_reward / max(root.explore_count, 1)
    draw_info = f"  draw={root.draw_rate:.3f}" if root.draw_rate > 0.001 else ""
    print(f"  value = {root_val:+.4f}{draw_info}    sims = {root.explore_count}"
          f"{'  (solved)' if root.outcome is not None else ''}")

    # Build combined: NN raw + MCTS side by side
    nn_policy = None
    if evaluator is not None:
        nn_value, nn_policy = evaluator._inference(state)
        if hasattr(nn_value, '__len__') and not isinstance(nn_value, float):
            # WDL output: (w, d, l)
            w, d, l = float(nn_value[0]), float(nn_value[1]), float(nn_value[2])
            nn_val = (w - l)
            nn_str = f"w={w:.3f} d={d:.3f} l={l:.3f}"
        else:
            nn_val = float(nn_value)
            nn_str = f"{nn_val:+.4f}"
        print(f"  ── NN raw {nn_str}    "
              f"MCTS value={root_val:+.4f}{draw_info}    "
              f"sims={root.explore_count} ──")

    # Gather all legal actions with both policies
    legal = state.legal_actions()
    rows = []
    for a in legal:
        mcts_v = 0.0
        mcts_n = 0
        mcts_solved = " "
        for c in root.children:
            if c.action == a:
                mcts_n = c.explore_count
                mcts_v = c.outcome[player] if c.outcome is not None else c.q_value
                mcts_solved = "✓" if c.outcome is not None else " "
                break
        nn_p = float(nn_policy[a]) if nn_policy is not None else 0.0
        if a >= 64:
            coord = "pass"
        else:
            row, col = a // COLS, a % COLS
            coord = f"({row},{col})"
        rows.append((a, coord, nn_p, mcts_n, mcts_v, mcts_solved))

    # Sort by MCTS visits, fallback to NN prior
    rows.sort(key=lambda r: (-r[3], -r[2]))

    total_visits = sum(c.explore_count for c in root.children)
    # Fixed-width columns: move(action) | NN prob + bar | MCTS prob + bar | V | N
    BAR_W = 30
    HDR = f"  {'move':>8s}  {'NN':>5s}  {'':<{BAR_W}s}  {'MCTS':>5s}  {'':<{BAR_W}s}  {'V':>7s}  {'N':>4s}"
    print(HDR)
    print("  " + "-" * (len(HDR) - 2))
    for a, coord, nn_p, mcts_n, mcts_v, mcts_solved in rows[:12]:
        nn_bar = "█" * int(nn_p * BAR_W) if nn_p > 0.001 else ""
        mcts_p = mcts_n / max(total_visits, 1)
        mcts_bar = "█" * int(mcts_p * BAR_W) if mcts_n > 0 else ""
        print(f"  {coord:>6s} {a:>3d}  {nn_p:.3f} {nn_bar:<{BAR_W}s}  "
              f"{mcts_p:.3f} {mcts_bar:<{BAR_W}s} {mcts_v:+.3f}{mcts_solved} {mcts_n:>4d}")
    print()

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


def draw_panel(screen, black_count, white_count, current_player, message="",
               value=None, draw_rate=0.0):
    """Draw info panel below the board."""
    small_font = pygame.font.Font(None, 26)
    font = pygame.font.Font(None, 30)
    y = HEIGHT + 8

    turn = "Black" if current_player == 0 else "White"
    texts = [
        f"Black: {black_count}    White: {white_count}    Turn: {turn}",
    ]
    if value is not None:
        # Convert to black perspective
        bq = value if current_player == 0 else -value
        if draw_rate > 0.001:
            bw = max(0.0, (bq + 1.0 - draw_rate) / 2.0)
            bl = max(0.0, (1.0 - bq - draw_rate) / 2.0)
            texts.append(f"Black W/D/L: {bw:.1%} / {draw_rate:.1%} / {bl:.1%}")
        else:
            texts.append(f"Value (black): {bq:+.3f}  ({((1+bq)*50):.1f}%)")
    texts.append(message)
    texts.append("H: hint   F: freeze   R: restart   Q: quit")

    for i, t in enumerate(texts):
        surf = small_font.render(t, True, BLACK)
        screen.blit(surf, (10, y))
        y += 26


def draw_hints_on_board(screen, hints, board_size=COLS):
    """Overlay MCTS visit counts and Q values on the board squares."""
    if not hints:
        return
    actions, visits, q_values = hints
    if not visits or not q_values:
        return
    max_visit = max(visits)
    best_q = max(q_values)
    font_q = pygame.font.Font(None, 30)
    font_v = pygame.font.Font(None, 30)

    for a, vst, q in zip(actions, visits, q_values):
        if a >= 64:
            continue
        row, col = a // board_size, a % board_size
        cx = col * SQ_SIZE + SQ_SIZE // 2
        cy = row * SQ_SIZE + SQ_SIZE // 2

        # Visit count — red if best, otherwise blue/dark
        if vst == max_visit and max_visit > 1:
            v_color = RED
        else:
            v_color = BLUE if vst > max_visit * 0.3 else GRAY
        v_txt = font_v.render(str(vst), True, v_color)
        screen.blit(v_txt, (cx - v_txt.get_width() // 2, cy + SQ_SIZE // 2 - 18))

        # Q value — blue if best, else dark
        if q == best_q:
            q_color = BLUE
        else:
            q_color = BLACK
        q_txt = font_q.render(f"{q:+.2f}", True, q_color)
        screen.blit(q_txt, (cx - q_txt.get_width() // 2, cy - SQ_SIZE // 2 + 5))


def choose_color(screen):
    """Menu: pick black/white/HvH.  Returns 0, 1, or -1 for HvH."""
    font = pygame.font.Font(None, 40)
    btn_black = pygame.Rect(150, 200, 300, 60)
    btn_white = pygame.Rect(150, 300, 300, 60)
    btn_hvh   = pygame.Rect(150, 400, 300, 60)

    while True:
        screen.fill(GREEN)
        title = font.render("Play as:", True, BLACK)
        screen.blit(title, (250, 110))

        pygame.draw.rect(screen, GRAY, btn_black)
        pygame.draw.rect(screen, GRAY, btn_white)
        pygame.draw.rect(screen, GRAY, btn_hvh)
        screen.blit(font.render("Black (first)", True, WHITE), (210, 212))
        screen.blit(font.render("White (second)", True, WHITE), (210, 312))
        screen.blit(font.render("Human vs Human", True, WHITE), (210, 412))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_black.collidepoint(event.pos):
                    return 0
                if btn_white.collidepoint(event.pos):
                    return 1
                if btn_hvh.collidepoint(event.pos):
                    return -1
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


def get_hints_from_root(root):
    """Extract (actions, visits, q_values) from an MCTS root node.
    Uses proven outcome when available, else visit-based Q."""
    actions = []
    visits = []
    q_values = []
    for c in root.children:
        actions.append(c.action)
        visits.append(c.explore_count)
        v = c.q_value
        if c.outcome is not None:
            v = c.outcome[c.player]
        q_values.append(v)
    return actions, visits, q_values


def main():
    game = pyspiel.load_game("othello")
    print("Loading model...")
    model = load_model(game)
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
        freeze_hint = False
        hints = None
        ai_value = None
        hint_draw_rate = 0.0
        hint_root = None
        hint_state = None
        message = ""
        restart_game = False
        hvh = (human_color == -1)

        while not state.is_terminal() and not restart_game:
            obs = state.observation_tensor(0)
            board = obs_to_board(obs)
            black_c, white_c = get_piece_counts(board)
            current = state.current_player()

            if hvh or current == human_color:
                # Human turn — auto-pass if only pass is available
                legal = state.legal_actions()
                if legal == [64]:
                    state.apply_action(64)
                    print_state(state)
                    evaluator.clear_cache()
                    message = "Human auto-pass"
                    continue

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
                                print_state(state)
                                evaluator.clear_cache()
                                hints = None
                                ai_value = None
                                hint_draw_rate = 0.0
                                hint_root = None
                                message = ""
                                break
                            else:
                                message = "Illegal move"
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_h:
                            showing_hints = not showing_hints
                            freeze_hint = False
                            if not showing_hints:
                                hints = None
                                ai_value = None
                                hint_draw_rate = 0.0
                                hint_root = None
                            else:
                                hint_root = None  # force fresh search
                        elif event.key == pygame.K_f and showing_hints and hint_root is not None:
                            freeze_hint = not freeze_hint
                            if freeze_hint:
                                print_mcts_info(hint_root, state, evaluator)
                                message = "Hint FROZEN"
                            else:
                                message = ""
                        elif event.key == pygame.K_r:
                            restart_game = True
                        elif event.key == pygame.K_q:
                            pygame.quit()
                            sys.exit()

                if showing_hints and not state.is_terminal() and not freeze_hint:
                    # Persistent incremental search — tree accumulates sims.
                    if hint_root is None or hint_state != str(state):
                        hint_cfg = MCTSConfig(
                            max_simulations=64, batch_size=MCTS_BATCH_SIZE,
                            uct_c=UCT_C,
                            policy_epsilon=0, verbose=False)
                        hint_mcts = BatchMCTS(game, hint_cfg, evaluator,
                                              random_state=np.random.RandomState())
                        hint_root = hint_mcts.mcts_search(state)
                        hint_state = str(state)
                    elif hint_root.visit_count < HINT_MAX_SIM:
                        hint_mcts.config.max_simulations = 64
                        hint_root = hint_mcts.mcts_search(state, root=hint_root)
                    hints = get_hints_from_root(hint_root)
                    if hint_root.outcome is not None:
                        ai_value = hint_root.outcome[current]
                        hint_draw_rate = hint_root.draw_rate
                    else:
                        ai_value = hint_root.total_reward / max(hint_root.explore_count, 1)
                        hint_draw_rate = hint_root.draw_rate
            else:
                # AI turn
                message = "AI thinking..."
                draw_board(screen, board, state.legal_actions(), hints)
                draw_hints_on_board(screen, hints)
                draw_panel(screen, black_c, white_c, current, message,
                           value=ai_value, draw_rate=hint_draw_rate)
                pygame.display.flip()

                # Run MCTS and display info
                root = bot.mcts_search(state)
                print_state(state)
                print_mcts_info(root, state, evaluator)

                action = root.best_child().action
                state.apply_action(action)
                evaluator.clear_cache()
                # Only auto-show hints if toggle is on
                if showing_hints:
                    hints = get_hints_from_root(root)
                    if root.outcome is not None:
                        ai_value = root.outcome[current]
                        hint_draw_rate = root.draw_rate
                    else:
                        ai_value = root.total_reward / max(root.explore_count, 1)
                        hint_draw_rate = root.draw_rate
                else:
                    hints = None
                    ai_value = None
                    hint_draw_rate = 0.0
                message = f"AI played: {state.action_to_string(current, action)}"

            # Render
            if not state.is_terminal():
                obs = state.observation_tensor(0)
                board = obs_to_board(obs)
                black_c, white_c = get_piece_counts(board)
                legal = state.legal_actions()
                draw_board(screen, board, legal, hints)
                draw_hints_on_board(screen, hints)
                draw_panel(screen, black_c, white_c, state.current_player(), message,
                           value=ai_value, draw_rate=hint_draw_rate)

            clock.tick(30)
            pygame.display.flip()

        # Game over (skip if restart was requested mid-game)
        if not restart_game:
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
