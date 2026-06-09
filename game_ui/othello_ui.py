"""Play Othello against a trained PyTorch AlphaZero model via OpenSpiel."""

import sys
import os
_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

import pygame

from game_ui.core import (load_model_for_ui, create_bot, MCTSHintEngine,
                          choose_color_menu, game_over_screen, print_mcts_info,
                          BLACK, WHITE, GRAY, RED, BLUE, GREEN, YELLOW)
from game_ui.othello_render import (
    ROWS, COLS, SQ_SIZE, BOARD_W, BOARD_H, PANEL_W, PADDING,
    draw_board_bg, draw_pieces, draw_legal_dots, draw_mcts_hints,
    obs_to_board, get_piece_counts, action_label, click_to_action)
from train.core.model_builder import build_othello_model

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = r"C:\Users\shouk\othello_train\cloud_wdl_mix"
CHECKPOINT_STEP = 125
MCTS_SIMULATIONS = 4096
HINT_MAX_SIM = 12800
MCTS_BATCH_SIZE = 16
UCT_C = 5

WIDTH = BOARD_W
HEIGHT = BOARD_H
PANEL_HEIGHT = 120
SCREEN_HEIGHT = HEIGHT + PANEL_HEIGHT


# ── UI helpers ──────────────────────────────────────────────────────────────

def draw_panel(screen, black_c, white_c, cur_player, message="",
               value=None, draw_rate=0.0):
    font_sm = pygame.font.Font(None, 26)
    y = HEIGHT + 8

    turn = "Black" if cur_player == 0 else "White"
    lines = [f"Black: {black_c}    White: {white_c}    Turn: {turn}"]

    if value is not None:
        bq = value if cur_player == 0 else -value
        if draw_rate > 0.001:
            bw = max(0.0, (bq + 1.0 - draw_rate) / 2.0)
            bl = max(0.0, (1.0 - bq - draw_rate) / 2.0)
            lines.append(f"Black W/D/L: {bw:.1%} / {draw_rate:.1%} / {bl:.1%}")
        else:
            lines.append(f"Value (black): {bq:+.3f}  ({((1+bq)*50):.1f}%)")
    lines.append(message)
    lines.append("H: hint   F: freeze   R: restart   Q: quit")

    for t in lines:
        surf = font_sm.render(t, True, BLACK)
        screen.blit(surf, (10, y))
        y += 26


def main():
    model, game = load_model_for_ui(CHECKPOINT_DIR, CHECKPOINT_STEP,
                                    build_othello_model)
    bot, evaluator, _ = create_bot(game, model, MCTS_SIMULATIONS,
                                   MCTS_BATCH_SIZE, UCT_C)
    hint_engine = MCTSHintEngine(game, evaluator, HINT_MAX_SIM,
                                 MCTS_BATCH_SIZE, UCT_C)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Othello vs AlphaZero")
    clock = pygame.time.Clock()

    while True:
        human_color = choose_color_menu(screen, WIDTH, HEIGHT)
        if human_color is None:
            break

        state = game.new_initial_state()
        hints = None
        ai_value = None
        hint_draw_rate = 0.0
        hint_frozen = False
        message = ""
        restart = False
        hvh = (human_color == -1)

        while not state.is_terminal() and not restart:
            obs = state.observation_tensor(0)
            board, _ = obs_to_board(obs)
            black_c, white_c = get_piece_counts(board)
            cur = state.current_player()

            if hvh or cur == human_color:
                legal = state.legal_actions()
                if legal == [64]:
                    state.apply_action(64)
                    evaluator.clear_cache()
                    message = "Human auto-pass"
                    continue

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        action = click_to_action(event.pos, legal)
                        if action is not None:
                            state.apply_action(action)
                            evaluator.clear_cache()
                            hints = None; ai_value = None
                            hint_draw_rate = 0.0
                            hint_engine.subtree_inherit(action)
                            message = ""
                        else:
                            message = "Illegal move"
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_h:
                            hint_frozen = False
                            if hints is None:
                                act, vst, qv, ai_val, dr = hint_engine.search(state)
                                hints = (act, vst, qv)
                                ai_value = ai_val; hint_draw_rate = dr
                            else:
                                hints = None; ai_value = None
                                hint_draw_rate = 0.0
                        elif event.key == pygame.K_f and hints is not None:
                            hint_engine.freeze()
                            hint_frozen = True
                            print_mcts_info(hint_engine._root, state,
                                           evaluator, action_label)
                            message = "Hint FROZEN"
                        elif event.key == pygame.K_r:
                            restart = True
                        elif event.key == pygame.K_q:
                            pygame.quit(); sys.exit()

                if hints is not None and not hint_frozen:
                    act, vst, qv, ai_val, dr = hint_engine.search(state)
                    hints = (act, vst, qv)
                    ai_value = ai_val; hint_draw_rate = dr
            else:
                message = "AI thinking..."
                screen.fill(GREEN)
                draw_board_bg(screen); draw_pieces(screen, board)
                if hints:
                    draw_mcts_hints(screen, *hints)
                draw_legal_dots(screen, state.legal_actions())
                draw_panel(screen, black_c, white_c, cur, message,
                           value=ai_value, draw_rate=hint_draw_rate)
                pygame.display.flip()

                policy, action = bot.step_with_policy(state)
                print_mcts_info(bot._last_root, state, evaluator, action_label)
                state.apply_action(action)
                evaluator.clear_cache()
                hint_engine.subtree_inherit(action)

                if hints is not None and not hint_frozen:
                    act, vst, qv, ai_val, dr = hint_engine.search(state)
                    hints = (act, vst, qv)
                    ai_value = ai_val; hint_draw_rate = dr
                else:
                    hints = None; ai_value = None; hint_draw_rate = 0.0
                message = f"AI played: {action_label(action)}"

            if not state.is_terminal():
                obs = state.observation_tensor(0)
                board, _ = obs_to_board(obs)
                black_c, white_c = get_piece_counts(board)
                screen.fill(GREEN)
                draw_board_bg(screen); draw_pieces(screen, board)
                draw_legal_dots(screen, state.legal_actions())
                if hints:
                    draw_mcts_hints(screen, *hints)
                draw_panel(screen, black_c, white_c, state.current_player(),
                           message, value=ai_value, draw_rate=hint_draw_rate)

            clock.tick(30)
            pygame.display.flip()

        if not restart:
            obs = state.observation_tensor(0)
            board, _ = obs_to_board(obs)
            black_c, white_c = get_piece_counts(board)

            if black_c > white_c:
                result = "Black wins!"
            elif white_c > black_c:
                result = "White wins!"
            else:
                result = "Draw!"

            if not game_over_screen(screen, WIDTH, HEIGHT, result,
                                    lambda scr, bd: (draw_board_bg(scr),
                                                     draw_pieces(scr, bd)),
                                    board):
                break

    pygame.quit()


if __name__ == "__main__":
    main()
