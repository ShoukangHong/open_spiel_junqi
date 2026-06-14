"""Play Xiangqi (Chinese Chess) against a trained AlphaZero model."""

import sys
import os
_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

import pygame

from game_ui.core import (load_model_for_ui, create_bot, MCTSHintEngine,
                          choose_color_menu, game_over_screen, print_mcts_info,
                          BLACK, WHITE, RED, BLUE, GREEN, YELLOW, get_font)
from game_ui.xiangqi_render import (
    ROWS, COLS, SQ_SIZE, LABEL_MARGIN, BOARD_W, BOARD_H, PANEL_W,
    BG_COLOR, LINE_COLOR, BoardData,
    draw_board_bg, draw_board_ui, draw_pieces, draw_legal_dots,
    draw_source_heatmap, draw_mcts_hints, draw_move_arrows, obs_to_board,
    action_label, click_to_square, ActionSelector, compute_source_probs,
    hint_arrows)
from train.core.model_builder import build_xiangqi_model

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = r"C:\Users\shouk\xiangqi_train\cloud"
CHECKPOINT_STEP = 30
MCTS_SIMULATIONS = 4096
HINT_MAX_SIM = 16000
MCTS_BATCH_SIZE = 32
UCT_C = 1.41
AI_TEMPERATURE = 0.1  # τ for AI move selection (0 = argmax)

WIDTH = BOARD_W
HEIGHT = BOARD_H
PANEL_HEIGHT = 80
SCREEN_HEIGHT = HEIGHT + PANEL_HEIGHT


# ── Panel ───────────────────────────────────────────────────────────────────

def draw_panel(screen, cur_player, message="", value=None, draw_rate=0.0):
    font_sm = get_font(22)
    y = HEIGHT + 8

    turn = "Red" if cur_player == 0 else "Black"
    lines = [f"Turn: {turn}"]
    if value is not None:
        bq = value if cur_player == 0 else -value
        if draw_rate > 0.001:
            bw = max(0.0, (bq + 1.0 - draw_rate) / 2.0)
            bl = max(0.0, (1.0 - bq - draw_rate) / 2.0)
            lines.append(f"Red W/D/L: {bw:.1%} / {draw_rate:.1%} / {bl:.1%}")
        else:
            lines.append(f"Value (red): {bq:+.3f}")
    lines.append(message)
    lines.append("H: hint   F: freeze   R: restart   Q: quit")

    for t in lines:
        surf = font_sm.render(t, True, BLACK)
        screen.blit(surf, (10, y))
        y += 26


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    try:
        model, game = load_model_for_ui(CHECKPOINT_DIR, CHECKPOINT_STEP,
                                        build_xiangqi_model)
    except FileNotFoundError:
        print(f"[ui] No checkpoint found at {CHECKPOINT_DIR}, using random model")
        import pyspiel
        game = pyspiel.load_game("xiangqi")
        model = build_xiangqi_model(game, {"nn_width": 32, "nn_depth": 5,
                                           "device": "cpu", "path": "."})

    bot, evaluator, _ = create_bot(game, model, MCTS_SIMULATIONS,
                                   MCTS_BATCH_SIZE, UCT_C)
    hint_engine = MCTSHintEngine(game, evaluator, HINT_MAX_SIM,
                                 MCTS_BATCH_SIZE, UCT_C)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Xiangqi vs AlphaZero")
    clock = pygame.time.Clock()

    while True:
        human_color = choose_color_menu(screen, WIDTH, HEIGHT)
        if human_color is None:
            break

        state = game.new_initial_state()
        selector = ActionSelector()
        hints = None
        ai_value = None
        hint_draw_rate = 0.0
        hint_src_probs = None
        _hint_arrows = None
        hint_frozen = False
        show_hints = False
        message = ""
        restart = False
        hvh = (human_color == -1)

        while not state.is_terminal() and not restart:
            board_data = obs_to_board(state.observation_tensor())
            cur = state.current_player()
            legal = state.legal_actions()

            if hvh or cur == human_color:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        action, src, tgt = selector.handle_click(event.pos, state)
                        if action is not None:
                            state.apply_action(action)
                            evaluator.clear_cache()
                            hint_engine.subtree_inherit(action)
                            if not show_hints:
                                hints = None; ai_value = None
                                hint_src_probs = None; _hint_arrows = None
                                hint_draw_rate = 0.0
                            else:
                                hints = None  # force re-search in hint update section
                            selector.reset()
                            message = f"Played: {action_label(action)}"
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_h:
                            hint_frozen = False
                            hint_engine.unfreeze()
                            show_hints = not show_hints
                            if not show_hints:
                                hints = None; ai_value = None
                                hint_src_probs = None; _hint_arrows = None
                                hint_draw_rate = 0.0
                        elif event.key == pygame.K_f and hints is not None:
                            hint_engine.freeze()
                            hint_frozen = True
                            if hint_engine._root is not None:
                                print_mcts_info(hint_engine._root, state,
                                               evaluator, action_label)
                            message = "Hint FROZEN"
                        elif event.key == pygame.K_r:
                            restart = True
                        elif event.key == pygame.K_q:
                            pygame.quit(); sys.exit()

                if show_hints and not hint_frozen:
                    act, vst, qv, ai_val, dr = hint_engine.search(state)
                    hints = (act, vst, qv)
                    ai_value = ai_val; hint_draw_rate = dr
                    src_visits = {}
                    for a, v in zip(act, vst):
                        sq = a // 90
                        src_visits[sq] = src_visits.get(sq, 0) + v
                    total = sum(src_visits.values()) or 1
                    hint_src_probs = {k: v/total for k, v in src_visits.items()}
                    _hint_arrows = hint_arrows(act, vst, hint_engine._root)
            else:
                message = "AI thinking..."
                screen.fill(BG_COLOR)
                draw_board_ui(screen, board_data, legal,
                             selected_src=selector.selected_src,
                             hint_data=hints, hint_heatmap=hint_src_probs,
                             move_arrows=_hint_arrows)
                draw_panel(screen, cur, message, value=ai_value,
                          draw_rate=hint_draw_rate)
                pygame.display.flip()

                policy, action = bot.step_with_policy(state, AI_TEMPERATURE)
                print_mcts_info(bot._last_root, state, evaluator, action_label)
                state.apply_action(action)
                evaluator.clear_cache()
                hint_engine.subtree_inherit(action)

                hint_engine.subtree_inherit(action)
                if not show_hints:
                    hints = None; ai_value = None
                    hint_src_probs = None; _hint_arrows = None
                    hint_draw_rate = 0.0
                else:
                    hints = None  # force re-search next human turn
                message = f"AI played: {action_label(action)}"

            if not state.is_terminal():
                board_data = obs_to_board(state.observation_tensor())
                screen.fill(BG_COLOR)
                draw_board_ui(screen, board_data, legal,
                             selected_src=selector.selected_src,
                             hint_data=hints, hint_heatmap=hint_src_probs,
                             move_arrows=_hint_arrows)
                draw_panel(screen, state.current_player(), message,
                          value=ai_value, draw_rate=hint_draw_rate)

            clock.tick(30)
            pygame.display.flip()

        if not restart:
            r = state.returns()
            if r[0] > 0:
                result = "Red wins!"
            elif r[0] < 0:
                result = "Black wins!"
            else:
                result = "Draw!"

            board_data = obs_to_board(state.observation_tensor(0))
            if not game_over_screen(screen, WIDTH, HEIGHT, result,
                                    lambda scr, bd: draw_board_ui(scr, bd),
                                    board_data):
                break

    pygame.quit()


if __name__ == "__main__":
    main()
