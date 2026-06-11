"""Visual browser for Xiangqi replay buffer (.npz / .db) files.

Usage:
    python buffer_viewer_xiangqi.py [buffer.db]

Controls:
    Left/Right  — +-1 sample           Up/Down   — +-10 samples
    PgUp/PgDn   — +-100 samples        Home/End  — first / last
    H           — toggle policy heatmap
    T           — cycle tag filter   R — filter rare only   A — show all
    Click       — select source square to see target heatmap
    Q / Esc     — quit
"""

import os
import sys

import numpy as np
import pygame

from game_ui.core import (
    load_buffer, TagFilter, build_filtered_indices, tag_display,
    TAG_COLORS, TAG_FILTERS, BLACK, WHITE, GRAY, LIGHT_GRAY, RED, YELLOW, BG,
    get_font)
from game_ui.xiangqi_render import (
    ROWS, COLS, SQ_SIZE, LABEL_MARGIN, BOARD_W, BOARD_H, PANEL_W, PADDING,
    obs_to_board, action_label, action_label_with_piece, build_display_data,
    draw_board_viewer, draw_board_bg, draw_pieces, draw_source_heatmap,
    draw_target_heatmap, draw_move_arrows, compute_source_probs,
    compute_top_moves)

# ── Config ──────────────────────────────────────────────────────────────────
BUFFER_FILE = r"C:\Users\shouk\xiangqi_train\buffer.db"


# ── Side panel ──────────────────────────────────────────────────────────────

def draw_side_panel(screen, index, total, global_idx, tag_filter_name,
                    disp, tag_filter_obj, tags_arr, selected_src=None):
    x0 = BOARD_W + PADDING
    y = PADDING
    w = PANEL_W - 2 * PADDING
    f = get_font(20)
    sf = get_font(15)

    def _line(text, color=BLACK, font=f, advance=True):
        nonlocal y
        s = font.render(text, True, color)
        screen.blit(s, (x0, y))
        if advance:
            y += font.get_height() + 2

    def _hline():
        nonlocal y
        pygame.draw.line(screen, LIGHT_GRAY, (x0, y), (x0 + w, y))
        y += 5

    # Sample info
    _line(f"Sample #{global_idx}  ({index + 1}/{total})")
    pname = "RED" if disp["cur_player"] == 0 else "BLACK"
    _line(f"Turn: {pname}    Pieces: {len(disp['pieces'])}")
    if disp["wdl"] is not None:
        w, d, l = disp["wdl"]
        _line(f"Value: {disp['value']:+.3f}  ({disp['win_pct']:.1f}%)")
        _line(f"  W/D/L: {w:.1%}/{d:.1%}/{l:.1%}")
    else:
        _line(f"Value: {disp['value']:+.3f}  ({disp['win_pct']:.1f}%)")

    _line("Tag: ", advance=False)
    x_after = x0 + f.render("Tag: ", True, BLACK).get_width()
    s = f.render(disp["tag_label"], True, disp["tag_color"])
    screen.blit(s, (x_after, y))
    y += f.get_height() + 2
    _line(f"Filter: [{tag_filter_name}]", color=GRAY)

    # Selected source info
    if selected_src is not None:
        _line(f"Source: {selected_src}", color=RED)
    _hline()

    # Top-5 policy
    _line("Top policy:")
    bar_max = 80
    bx0 = x0 + 100
    pieces = disp["pieces"]
    for a, p, legal in disp["top5"]:
        label = action_label_with_piece(a, pieces) if pieces else action_label(a)
        clr = BLACK if legal else GRAY
        # Truncate long labels
        if len(label) > 14:
            label = label[:13] + ".."
        screen.blit(sf.render(f"  {label:<15s}", True, clr), (x0, y))
        bw = int(p * bar_max)
        bar_clr = (200, 50, 50) if p > 0.1 else (140, 140, 140)
        if bw > 0:
            pygame.draw.rect(screen, bar_clr, (bx0, y + 1, bw, 8))
        screen.blit(sf.render(f" {p:.3f}", True, BLACK), (bx0 + bw + 3, y))
        y += sf.get_height() + 1
    _hline()

    # Browse
    _line("Browse:")
    n_show = 10
    half = n_show // 2
    f_idx = tag_filter_obj.indices
    total_f = len(f_idx)
    if total_f == 0:
        _line("  (no entries)", color=GRAY, font=sf)
    else:
        start = max(0, min(index - half, total_f - n_show))
        end = min(total_f, start + n_show)
        for gi in f_idx[start:end]:
            gi_int = int(gi)
            is_current = (gi_int == global_idx)
            marker = ">" if is_current else " "
            label_txt = f"{marker} {gi_int:>5d}"
            t = str(tags_arr[gi_int]) if tags_arr is not None else ""
            td = tag_display(t)
            tag_clr = TAG_COLORS.get(td, BLACK)
            clr = WHITE if is_current else BLACK
            bg_clr = (100, 140, 200) if is_current else None
            if bg_clr is not None:
                pygame.draw.rect(screen, bg_clr, (x0, y, w, sf.get_height() + 2))
            screen.blit(sf.render(label_txt, True, clr), (x0 + 2, y + 1))
            if td != "normal":
                tw = sf.render(label_txt, True, clr).get_width()
                screen.blit(sf.render(f"[{td}]", True, tag_clr),
                            (x0 + tw + 10, y + 1))
            y += sf.get_height() + 2
    _hline()

    for line_text in [
        "<- -> : +-1     up/down : +-10",
        "PgUp/Dn: +-100   Home/End",
        "H: heatmap   Click: select src",
        "T: cycle filter   R: rare   A: all",
        "Q/Esc: quit",
    ]:
        screen.blit(sf.render(line_text, True, GRAY), (x0, y))
        y += sf.get_height() + 1


# ── Main ────────────────────────────────────────────────────────────────────

WIDTH = BOARD_W + PANEL_W
HEIGHT = BOARD_H

def main():
    path = BUFFER_FILE
    if len(sys.argv) > 1:
        path = sys.argv[1]

    import sys as _sys
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    obs_arr, masks_arr, policies_arr, values_arr, tags_arr = load_buffer(path)
    total_global = len(values_arr)
    if tags_arr is not None:
        unique, counts = np.unique(tags_arr, return_counts=True)
        tc = {str(k): int(v) for k, v in zip(unique, counts)}
        print(f"Loaded {total_global}  tags={tc}")
    else:
        print(f"Loaded {total_global} (no tags)")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Xiangqi Buffer Viewer — {os.path.basename(path)}")
    clock = pygame.time.Clock()

    tag_filter = TagFilter(tags_arr, "all")
    show_heatmap = True
    selected_src = None   # (row, col) of selected source square
    _disp = None
    _last_gi = -1
    _policy_cache = None  # cached policy for target heatmap
    _top_moves = []       # cached top-N moves for arrows
    _mask_cache = None    # cached mask for compute_top_moves

    def refresh_disp():
        nonlocal _disp, _last_gi, _policy_cache, _top_moves, _mask_cache
        idx = tag_filter.indices
        if len(idx) == 0:
            _disp = None; return
        pos = tag_filter.pos % len(idx)
        g_idx = int(idx[pos])
        tag = tags_arr[g_idx] if tags_arr is not None else None
        board_data = obs_to_board(obs_arr[g_idx])
        policy = policies_arr[g_idx]
        mask = masks_arr[g_idx]
        _policy_cache = policy
        _mask_cache = mask
        _top_moves = compute_top_moves(policy, mask, top_n=8)
        _disp = build_display_data(
            board_data, policy, mask, values_arr[g_idx], tag,
            board_data.cur_player)
        _last_gi = g_idx

    refresh_disp()

    running = True
    while running:
        idx = tag_filter.indices
        if len(idx) == 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_a:
                        tag_filter.set_filter("all"); refresh_disp()
                    elif event.key == pygame.K_t:
                        cur = (TAG_FILTERS.index(tag_filter.name) + 1) % len(TAG_FILTERS)
                        tag_filter.set_filter(TAG_FILTERS[cur]); refresh_disp()
            screen.fill(BG)
            pygame.display.flip()
            clock.tick(30)
            continue

        need_refresh = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                sq = (event.pos[1] // SQ_SIZE, (event.pos[0] - LABEL_MARGIN) // SQ_SIZE)
                if 0 <= sq[0] < ROWS and 0 <= sq[1] < COLS:
                    if selected_src == sq:
                        selected_src = None
                    else:
                        selected_src = sq
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_RIGHT: tag_filter.next(1); need_refresh = True
                elif key == pygame.K_LEFT: tag_filter.next(-1); need_refresh = True
                elif key == pygame.K_UP: tag_filter.next(10); need_refresh = True
                elif key == pygame.K_DOWN: tag_filter.next(-10); need_refresh = True
                elif key == pygame.K_PAGEUP: tag_filter.next(100); need_refresh = True
                elif key == pygame.K_PAGEDOWN: tag_filter.next(-100); need_refresh = True
                elif key == pygame.K_HOME: tag_filter.goto_first(); need_refresh = True
                elif key == pygame.K_END: tag_filter.goto_last(); need_refresh = True
                elif key == pygame.K_h: show_heatmap = not show_heatmap
                elif key == pygame.K_r: tag_filter.set_filter("rare"); need_refresh = True
                elif key == pygame.K_f: tag_filter.set_filter("rare_flip"); need_refresh = True
                elif key == pygame.K_t:
                    cur = (TAG_FILTERS.index(tag_filter.name) + 1) % len(TAG_FILTERS)
                    tag_filter.set_filter(TAG_FILTERS[cur]); need_refresh = True
                elif key == pygame.K_a: tag_filter.set_filter("all"); need_refresh = True
                elif key in (pygame.K_q, pygame.K_ESCAPE): running = False

        if need_refresh:
            selected_src = None
            refresh_disp()

        screen.fill(BG)
        if _disp is not None and len(tag_filter.indices) > 0:
            draw_board_viewer(screen, _disp, show_heatmap)

            # Move arrows
            if selected_src is not None and _policy_cache is not None:
                src_sq = selected_src[0] * COLS + selected_src[1]
                draw_target_heatmap(screen, src_sq, _policy_cache)
                # Arrows only for selected source
                src_moves = [(f, t, w) for f, t, w in _top_moves if f == src_sq]
                draw_move_arrows(screen, src_moves[:5])
                # Highlight selected square
                ox = LABEL_MARGIN
                cx = ox + selected_src[1] * SQ_SIZE + SQ_SIZE // 2
                cy = selected_src[0] * SQ_SIZE + SQ_SIZE // 2
                pygame.draw.circle(screen, YELLOW, (cx, cy), SQ_SIZE // 2 - 2, 3)
            else:
                # Show top global moves as arrows
                draw_move_arrows(screen, _top_moves[:5])

            draw_side_panel(screen, tag_filter.pos, len(tag_filter.indices),
                           tag_filter.current_global_idx, tag_filter.name,
                           _disp, tag_filter, tags_arr, selected_src)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
