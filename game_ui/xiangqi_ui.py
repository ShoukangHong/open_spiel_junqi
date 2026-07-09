"""Play Xiangqi (Chinese Chess) against a trained AlphaZero model."""

import os
import sys

_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

import numpy as np
import pygame

from game_ui.core import (load_model_for_ui, create_bot, create_ab_bot,
                          MCTSHintEngine, AlphaBetaHintEngine,
                          choose_color_menu, game_over_screen, print_search_info,
                          BLACK, get_font)
from game_ui.xiangqi_render import (
    BOARD_W, BOARD_H, BG_COLOR, draw_board_ui, obs_to_board,
    action_label, ActionSelector)
from train.core.model_builder import build_xiangqi_model

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = r"C:\Users\shouk\xiangqi_train\cloud_buf"
CHECKPOINT_STEP = 5
MCTS_SIMULATIONS = 1000
HINT_MAX_SIM = 10000
INFERENCE_BATCH_SIZE = 10  # shared by MCTS and AlphaBeta
UCT_C = 4.0
FPU_LAMBDA = 0.2  # FPU penalty for unvisited nodes in MCTS (0 = disabled)
PROBE_DEPTH = 1   # speculative probe layers (0 = disabled)
PROBE_SURPRISE = 1.0  # Q-drop threshold for probe early termination
AI_TEMPERATURE = 0.01  # τ for AI move selection (0 = argmax)
TEMP_DROP = 1         # use τ=0.5 + advantage mixing before this move
SAVE_DIR = os.path.join(CHECKPOINT_DIR, "saved_positions")
OPENING_DIR = os.path.join(CHECKPOINT_DIR, "opening_book")
POLICY_EPSILON = 0.0  # Dirichlet noise weight for AI/hint search
POLICY_ALPHA = 0.25     # Dirichlet concentration parameter
AB_DEPTH = 3           # alpha-beta search depth
AB_POLICY_TEMP = 0.003   # temperature for value→policy softmax in AB
_AB_FLAG = [False]      # toggle with 'A' key — list to allow mutation from nested scope
MAX_PRINT_MOVES = 25   # top N moves printed to console

# Read MCTS params from training config (fall back to defaults)
import json as _json, os as _os
_config_path = _os.path.join(CHECKPOINT_DIR, "train_config.json")
_tc = {}
if _os.path.exists(_config_path):
    with open(_config_path) as _f:
        _tc = _json.load(_f)
DRAW_PENALTY = _tc.get("draw_penalty", 0.3)
REPEAT_PENALTY = _tc.get("repeat_penalty", 0.1)

WIDTH = BOARD_W
HEIGHT = BOARD_H
PANEL_HEIGHT = 80
SCREEN_HEIGHT = HEIGHT + PANEL_HEIGHT


# ── Panel ───────────────────────────────────────────────────────────────────

def draw_panel(screen, cur_player, message="", value=None, draw_rate=0.0,
               move_num=0, no_cap=0):
    font_sm = get_font(22)
    y = HEIGHT + 8

    turn = "Red" if cur_player == 0 else "Black"
    lines = [f"Turn: {turn}"]
    lines.append(f"Move: {move_num}  No-cap: {no_cap}/40")
    if value is not None:
        bq = value if cur_player == 0 else -value
        if draw_rate > 0.001:
            bw = max(0.0, (bq + 1.0 - draw_rate) / 2.0)
            bl = max(0.0, (1.0 - bq - draw_rate) / 2.0)
            lines.append(f"Red W/D/L: {bw:.1%} / {draw_rate:.1%} / {bl:.1%}")
        else:
            lines.append(f"Value (red): {bq:+.3f}")
    lines.append(message)
    lines.append("H: hint  U: undo  F: freeze  A: AB/MCTS  S: save  D: opening  L: load  R: restart  Q: quit")

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

    def _make_bot_and_hint():
        if _AB_FLAG[0]:
            bot, evaluator = create_ab_bot(game, model, depth=AB_DEPTH,
                                            batch_size=INFERENCE_BATCH_SIZE,
                                            policy_temp=AB_POLICY_TEMP)
            hint_engine = AlphaBetaHintEngine(game, evaluator, depth=AB_DEPTH,
                                              batch_size=INFERENCE_BATCH_SIZE,
                                              policy_temp=AB_POLICY_TEMP)
        else:
            bot, evaluator, _ = create_bot(game, model, MCTS_SIMULATIONS,
                                           INFERENCE_BATCH_SIZE, UCT_C,
                                           DRAW_PENALTY, REPEAT_PENALTY,
                                           random_state=None,
                                           policy_epsilon=POLICY_EPSILON,
                                           policy_alpha=POLICY_ALPHA,
                                           fpu_lambda=FPU_LAMBDA,
                                           probe_depth=PROBE_DEPTH,
                                           probe_surprise=PROBE_SURPRISE)
            hint_engine = MCTSHintEngine(game, evaluator, HINT_MAX_SIM,
                                         INFERENCE_BATCH_SIZE, UCT_C,
                                         DRAW_PENALTY, REPEAT_PENALTY,
                                         POLICY_EPSILON, POLICY_ALPHA,
                                         fpu_lambda=FPU_LAMBDA,
                                         probe_depth=PROBE_DEPTH,
                                         probe_surprise=PROBE_SURPRISE)
        return bot, evaluator, hint_engine

    bot, evaluator, hint_engine = _make_bot_and_hint()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Xiangqi vs AlphaZero")
    clock = pygame.time.Clock()

    while True:
        human_color = choose_color_menu(screen, WIDTH, HEIGHT)
        if human_color is None:
            break

        state = game.new_initial_state()
        undo_stack = []  # state clones before each action
        selector = ActionSelector()
        hints = None; ai_value = None; _hint_printed = False
        hint_draw_rate = 0.0
        hint_src_probs = None
        _hint_arrows = None
        hint_frozen = False
        _last_dir = [None]  # mutable to update from event handler
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
                            if show_hints and hint_engine._root is not None:
                                hint_pol = hint_engine.compute_root_policy(
                                    hint_engine._root, state)
                                print_search_info(hint_engine._root, state, hint_pol,
                                               evaluator=evaluator, action_label_fn=action_label,
                                               max_moves=MAX_PRINT_MOVES,
                                               engine_label="AB" if _AB_FLAG[0] else "MCTS")
                            undo_stack.append(state.clone())
                            state.apply_action(action)
                            evaluator.clear_cache()
                            hint_engine.subtree_inherit(action)
                            if not show_hints:
                                hints = None; ai_value = None; _hint_printed = False
                                hint_src_probs = None; _hint_arrows = None
                                hint_draw_rate = 0.0
                                _hint_printed = False
                            else:
                                hints = None  # force re-search; _hint_printed = False
                            selector.reset()
                            message = f"Played: {action_label(action)}"
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_u and undo_stack:
                            steps = 2 if not hvh else 1
                            for _ in range(min(steps, len(undo_stack))):
                                state = undo_stack.pop()
                            evaluator.clear_cache()
                            _, evaluator, hint_engine = _make_bot_and_hint()
                            hints = None; ai_value = None; _hint_printed = False
                            hint_src_probs = None; _hint_arrows = None
                            hint_draw_rate = 0.0
                            selector.reset()
                            message = "Move undone"
                        elif event.key == pygame.K_h:
                            hint_frozen = False
                            hint_engine.unfreeze()
                            show_hints = not show_hints
                            if not show_hints:
                                hints = None; ai_value = None; _hint_printed = False
                                hint_src_probs = None; _hint_arrows = None
                                hint_draw_rate = 0.0
                                _hint_printed = False
                        elif event.key == pygame.K_f and hints is not None:
                            hint_engine.freeze()
                            hint_frozen = True
                            if hint_engine._root is not None:
                                hint_pol = hint_engine.compute_root_policy(
                                    hint_engine._root, state)
                                print_search_info(hint_engine._root, state, hint_pol,
                                               evaluator=evaluator, action_label_fn=action_label,
                                               max_moves=MAX_PRINT_MOVES,
                                               engine_label="AB" if _AB_FLAG[0] else "MCTS")
                            message = "Hint FROZEN"
                        elif event.key == pygame.K_s:
                            from tkinter import Tk; from tkinter.filedialog import asksaveasfilename
                            Tk().withdraw()
                            os.makedirs(SAVE_DIR, exist_ok=True)
                            import time as _time
                            defname = _time.strftime("xiangqi_%Y%m%d_%H%M%S.txt")
                            init_dir = _last_dir[0] if _last_dir[0] else SAVE_DIR
                            fpath = asksaveasfilename(
                                initialdir=init_dir, initialfile=defname,
                                defaultextension=".txt",
                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
                            if fpath:
                                with open(fpath, "w") as _f:
                                    _f.write(state.serialize())
                                _last_dir[0] = os.path.dirname(fpath)
                                message = f"Saved: {os.path.basename(fpath)}"
                            else:
                                message = "Save cancelled"
                        elif event.key == pygame.K_d:
                            os.makedirs(OPENING_DIR, exist_ok=True)
                            # Build filename from move coordinates
                            parts = []
                            for a in state.history():
                                fr, fc = divmod(a // 90, 9)
                                tr, tc = divmod(a % 90, 9)
                                parts.append(f"{fr}{fc}{tr}{tc}")
                            fname = "-".join(parts) + ".txt" if parts else "initial.txt"
                            fpath = os.path.join(OPENING_DIR, fname)
                            with open(fpath, "w") as _f:
                                _f.write(state.serialize())
                            message = f"Opening saved: {fname}"
                        elif event.key == pygame.K_l:
                            from tkinter import Tk; from tkinter.filedialog import askopenfilename
                            Tk().withdraw()
                            os.makedirs(SAVE_DIR, exist_ok=True)
                            init_dir = _last_dir[0] if _last_dir[0] else SAVE_DIR
                            fpath = askopenfilename(
                                initialdir=init_dir,
                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
                            if fpath:
                                with open(fpath) as _f:
                                    state = game.deserialize_state(_f.read().strip())
                                undo_stack.clear()
                                evaluator.clear_cache()
                                _, evaluator, hint_engine = _make_bot_and_hint()
                                hints = None; ai_value = None; _hint_printed = False
                                hint_src_probs = None; _hint_arrows = None
                                hint_draw_rate = 0.0
                                selector.reset()
                                _last_dir[0] = os.path.dirname(fpath)
                                message = f"Loaded: {os.path.basename(fpath)}"
                            else:
                                message = "Load cancelled"
                        elif event.key == pygame.K_a:
                            _AB_FLAG[0] = not _AB_FLAG[0]
                            bot, evaluator, hint_engine = _make_bot_and_hint()
                            hints = None; ai_value = None; _hint_printed = False
                            hint_src_probs = None; _hint_arrows = None
                            hint_draw_rate = 0.0
                            selector.reset()
                            mode = "Alpha-Beta" if _AB_FLAG[0] else "MCTS"
                            message = f"Switched to {mode}"
                        elif event.key == pygame.K_r:
                            restart = True
                        elif event.key == pygame.K_q:
                            pygame.quit(); sys.exit()

                if show_hints and not hint_frozen:
                    act, vst, qv, ai_val, dr = hint_engine.search(state)
                    hints = (act, vst, qv)
                    ai_value = ai_val; hint_draw_rate = dr
                    if hint_engine._root is not None:
                        hint_pol = hint_engine.compute_root_policy(
                            hint_engine._root, state)
                        if not _hint_printed:
                            print_search_info(hint_engine._root, state, hint_pol,
                                             evaluator=evaluator, action_label_fn=action_label,
                                             max_moves=MAX_PRINT_MOVES,
                                             engine_label="AB" if _AB_FLAG[0] else "MCTS")
                            _hint_printed = True
                        # Source heatmap from policy (not uniform visits)
                        src_visits = {}
                        for a in act:
                            sq = a // 90
                            src_visits[sq] = src_visits.get(sq, 0) + hint_pol.get(a, 0)
                        total = sum(src_visits.values()) or 1
                        hint_src_probs = {k: v/total for k, v in src_visits.items()}
                        # Arrows from policy
                        pol_list = [(a // 90, a % 90, hint_pol.get(a, 0)) for a in act]
                        pol_list.sort(key=lambda x: -x[2])
                        _hint_arrows = pol_list[:5]
                    else:
                        _hint_arrows = None
                        hint_src_probs = None
            else:
                message = "AI thinking..."
                screen.fill(BG_COLOR)
                draw_board_ui(screen, board_data, legal,
                             selected_src=selector.selected_src,
                             hint_data=hints, hint_heatmap=hint_src_probs,
                             move_arrows=_hint_arrows)
                obs_a = np.asarray(state.observation_tensor(), dtype=np.float32)
                mn = state.move_number()
                nc = int(obs_a[16 * 10 * 9] * 40 + 0.5)
                draw_panel(screen, cur, message, value=ai_value,
                          draw_rate=hint_draw_rate, move_num=mn, no_cap=nc)
                pygame.display.flip()

                mn = state.move_number()
                if mn < TEMP_DROP:
                    from train.core.policy import select_action_with_adv
                    root = bot.mcts_search(state)
                    policy = bot.compute_root_policy(root, state)
                    action, probs = select_action_with_adv(
                        root, state, temperature=2/3, alpha=0.3, adv_t=0.2,
                        base_policy=policy)
                    bot._last_root = root
                    # Display sharpened policy, not raw
                    display_pol = {c.action: float(probs[c.action])
                                   for c in root.children}
                    print_search_info(root, state, display_pol,
                                    evaluator=evaluator, action_label_fn=action_label,
                                    max_moves=MAX_PRINT_MOVES,
                                    engine_label="AB" if _AB_FLAG[0] else "MCTS")
                else:
                    from train.core.policy import select_action_with_adv
                    root = bot.mcts_search(state)
                    policy = bot.compute_root_policy(root, state)
                    action, probs = select_action_with_adv(
                        root, state, temperature=AI_TEMPERATURE, alpha=0.0,
                        base_policy=policy)
                    bot._last_root = root
                    display_pol = {c.action: float(probs[c.action])
                                   for c in root.children}
                    print_search_info(root, state, display_pol,
                               evaluator=evaluator, action_label_fn=action_label,
                               max_moves=MAX_PRINT_MOVES,
                               engine_label="AB" if _AB_FLAG[0] else "MCTS")
                undo_stack.append(state.clone())
                state.apply_action(action)
                evaluator.clear_cache()
                hint_engine.subtree_inherit(action)

                hint_engine.subtree_inherit(action)
                if not show_hints:
                    hints = None; ai_value = None; _hint_printed = False
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
                obs_p = np.asarray(state.observation_tensor(), dtype=np.float32)
                mn = state.move_number()
                nc = int(obs_p[16 * 10 * 9] * 40 + 0.5)
                draw_panel(screen, state.current_player(), message,
                          value=ai_value, draw_rate=hint_draw_rate,
                          move_num=mn, no_cap=nc)

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
