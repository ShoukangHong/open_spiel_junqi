"""Othello board rendering — shared between game_ui and buffer_viewer."""

import numpy as np
import pygame

from game_ui.core import (BLACK, WHITE, GRAY, RED, BLUE, GREEN,
                          BG, get_font, heatmap_surface)

# ── Layout constants ────────────────────────────────────────────────────────

ROWS = COLS = 8
SQ_SIZE = 80
LABEL_MARGIN = 20
BOARD_W = LABEL_MARGIN + COLS * SQ_SIZE   # 660
BOARD_H = ROWS * SQ_SIZE                   # 640
PANEL_W = 290
PADDING = 10

DARK_GREEN = (0, 100, 0)
YELLOW = (255, 255, 0)

# ── Observation parsing ─────────────────────────────────────────────────────

def obs_to_board(obs):
    """Convert [4, 8, 8] observation to (board_flat, cur_player).
    board: 1=black, -1=white, 0=empty."""
    obs = np.asarray(obs, dtype=np.float32).reshape(4, 8, 8)
    cur = 0 if obs[3, 0, 0] == 1 else 1
    return (obs[1] - obs[2]).flatten().astype(int), cur


def get_piece_counts(board):
    return int((board == 1).sum()), int((board == -1).sum())


# ── Action labels ───────────────────────────────────────────────────────────

def action_label(action):
    """Convert Othello action index to human-readable string."""
    if action >= 64:
        return "pass"
    return f"{chr(ord('a') + action % 8)}{action // 8 + 1}"


# ── Input ───────────────────────────────────────────────────────────────────

def click_to_action(pos, legal_actions):
    """Convert a mouse click to an action index, or None if illegal."""
    x, y = pos
    col = (x - LABEL_MARGIN) // SQ_SIZE
    row = y // SQ_SIZE
    if 0 <= row < ROWS and 0 <= col < COLS:
        action = row * COLS + col
        if action in legal_actions:
            return action
    return None


# ── Board drawing ───────────────────────────────────────────────────────────

def draw_board_bg(screen):
    """Green board + grid lines + row/col labels."""
    ox = LABEL_MARGIN
    board_h = ROWS * SQ_SIZE

    pygame.draw.rect(screen, GREEN, (ox, 0, COLS * SQ_SIZE, board_h))

    for i in range(COLS + 1):
        x = ox + i * SQ_SIZE
        pygame.draw.line(screen, BLACK, (x, 0), (x, board_h), 2)
    for i in range(ROWS + 1):
        pygame.draw.line(screen, BLACK, (ox, i * SQ_SIZE),
                         (ox + COLS * SQ_SIZE, i * SQ_SIZE), 2)

    f = get_font(18)
    for i in range(ROWS):
        t = f.render(str(i + 1), True, BLACK)
        screen.blit(t, (2, i * SQ_SIZE + SQ_SIZE // 2 - t.get_height() // 2))
    for i in range(COLS):
        t = f.render(chr(ord('a') + i), True, BLACK)
        screen.blit(t, (ox + i * SQ_SIZE + SQ_SIZE // 2 - t.get_width() // 2,
                        board_h + 2))


def draw_pieces(screen, board):
    """Black/white circles on the board."""
    ox = LABEL_MARGIN
    for row in range(ROWS):
        for col in range(COLS):
            v = board[row * COLS + col]
            if v == 1:
                cx = ox + col * SQ_SIZE + SQ_SIZE // 2
                cy = row * SQ_SIZE + SQ_SIZE // 2
                pygame.draw.circle(screen, BLACK, (cx, cy), SQ_SIZE // 2 - 3)
            elif v == -1:
                cx = ox + col * SQ_SIZE + SQ_SIZE // 2
                cy = row * SQ_SIZE + SQ_SIZE // 2
                pygame.draw.circle(screen, WHITE, (cx, cy), SQ_SIZE // 2 - 3)


def draw_legal_dots(screen, legal_actions):
    """Small dots indicating legal moves."""
    ox = LABEL_MARGIN
    for a in legal_actions:
        if a >= 64:
            continue
        row, col = a // COLS, a % COLS
        cx = ox + col * SQ_SIZE + SQ_SIZE // 2
        cy = row * SQ_SIZE + SQ_SIZE // 2
        pygame.draw.circle(screen, DARK_GREEN, (cx, cy), 8)


def draw_heatmap(screen, heatmap_cells):
    """Overlay heatmap on board squares.  *heatmap_cells*: list of (action, intensity)."""
    ox = LABEL_MARGIN
    for a, intensity in heatmap_cells:
        row, col = a // COLS, a % COLS
        s = heatmap_surface(SQ_SIZE, intensity)
        screen.blit(s, (ox + col * SQ_SIZE, row * SQ_SIZE))


def draw_policy_hints(screen, hints):
    """Overlay probability numbers.  *hints*: list of (action, prob, color, size)."""
    ox = LABEL_MARGIN
    for a, p, clr, size in hints:
        row, col = a // COLS, a % COLS
        cx = ox + col * SQ_SIZE + SQ_SIZE // 2
        cy = row * SQ_SIZE + SQ_SIZE // 2
        font = get_font(24 if size == "big" else 20)
        t = font.render(f"{p:.2f}", True, clr)
        screen.blit(t, (cx - t.get_width() // 2,
                        cy + SQ_SIZE // 2 - t.get_height() - 2))


def draw_mcts_hints(screen, actions, visits, q_values):
    """Overlay MCTS visit counts and Q values on the board."""
    if not actions or not visits or not q_values:
        return
    ox = LABEL_MARGIN
    max_visit = max(visits)
    best_q = max(q_values)
    font_q = get_font(30)
    font_v = get_font(30)

    for a, vst, q in zip(actions, visits, q_values):
        if a >= 64:
            continue
        row, col = a // COLS, a % COLS
        cx = ox + col * SQ_SIZE + SQ_SIZE // 2
        cy = row * SQ_SIZE + SQ_SIZE // 2

        v_color = RED if (vst == max_visit and max_visit > 1) else (
            BLUE if vst > max_visit * 0.3 else GRAY)
        v_txt = font_v.render(str(vst), True, v_color)
        screen.blit(v_txt, (cx - v_txt.get_width() // 2, cy + SQ_SIZE // 2 - 18))

        q_color = BLUE if q == best_q else BLACK
        q_txt = font_q.render(f"{q:+.2f}", True, q_color)
        screen.blit(q_txt, (cx - q_txt.get_width() // 2, cy - SQ_SIZE // 2 + 5))


# ── Combined draw (for buffer viewer) ───────────────────────────────────────

def build_display_data(board, policy, legal_mask, value, tag, cur_player):
    """Pre-compute everything needed for rendering one buffer sample."""
    d = {}
    d["board"] = board
    d["cur_player"] = cur_player
    d["black_c"] = int((board == 1).sum())
    d["white_c"] = int((board == -1).sum())

    v = np.asarray(value, dtype=np.float32).ravel()
    if len(v) == 3:
        d["wdl"] = (float(v[0]), float(v[1]), float(v[2]))
        d["value"] = float(v[0] - v[2])
    else:
        d["wdl"] = None
        d["value"] = float(v[0])
    d["win_pct"] = (1 + d["value"]) * 50

    from game_ui.core import tag_display, TAG_COLORS
    d["tag_label"] = tag_display(tag)
    d["tag_color"] = TAG_COLORS.get(d["tag_label"], BLACK)

    # Policy hints
    hints = []
    if policy is not None and legal_mask is not None:
        legal_actions = [a for a in range(65) if legal_mask[a]]
        best_action = max(legal_actions, key=lambda a: policy[a]) if legal_actions else -1
        for a in range(64):
            p = policy[a]
            if p < 0.005:
                continue
            if a == best_action and p > 0.01:
                clr = RED
                size = "big"
            elif legal_mask[a]:
                clr = BLUE
                size = "big"
            else:
                clr = GRAY
                size = "mid"
            hints.append((a, p, clr, size))
    d["hints"] = hints

    # Top-5
    if policy is not None:
        top5 = sorted([(a, policy[a]) for a in range(65)], key=lambda x: -x[1])[:5]
        d["top5"] = [(a, p, legal_mask is not None and legal_mask[a])
                      for a, p in top5]
    else:
        d["top5"] = []

    # Heatmap cells
    heatmap_cells = []
    if policy is not None:
        for a in range(64):
            p = policy[a]
            if p > 0.001:
                heatmap_cells.append((a, min(p * 3, 1.0)))
    d["heatmap_cells"] = heatmap_cells

    return d


def draw_board(screen, disp, show_heatmap=True):
    """Combined Othello board render for buffer viewer."""
    draw_board_bg(screen)
    if show_heatmap:
        draw_heatmap(screen, disp["heatmap_cells"])
    draw_pieces(screen, disp["board"])
    draw_policy_hints(screen, disp["hints"])