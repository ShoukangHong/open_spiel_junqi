"""Xiangqi board rendering — shared between game_ui and buffer_viewer.

Board: 10 rows × 9 cols, palace markings, river, Chinese character pieces.
Policy visualization: source-aggregated heatmap + per-source target heatmap.
Human input: two-click selection (source → target).
"""

import numpy as np
import pygame

from game_ui.core import (BLACK, WHITE, GRAY, RED, BLUE, GREEN,
                          YELLOW, get_font, heatmap_surface, tag_display,
                          TAG_COLORS)

# ── Layout constants ────────────────────────────────────────────────────────

ROWS = 10
COLS = 9
SQ_SIZE = 70
LABEL_MARGIN = 24
BOARD_W = LABEL_MARGIN + COLS * SQ_SIZE   # 654
BOARD_H = ROWS * SQ_SIZE                   # 700
PANEL_W = 320
PADDING = 10

BG_COLOR = (255, 248, 220)   # warm wood-like
LINE_COLOR = (60, 40, 20)    # dark brown

# ── Piece glyphs ────────────────────────────────────────────────────────────

# Red (player 0) and Black (player 1) — indexed by OpenSpiel piece type (1-7)
_RED_PIECES   = {1: "帥", 2: "仕", 3: "相", 4: "傌", 5: "俥", 6: "炮", 7: "兵"}
_BLACK_PIECES = {1: "將", 2: "士", 3: "象", 4: "馬", 5: "車", 6: "砲", 7: "卒"}
_PIECE_COLORS = {0: (200, 0, 0), 1: (0, 0, 0)}   # red, black

_PIECE_FONT = None

def _ensure_piece_font():
    global _PIECE_FONT
    if _PIECE_FONT is None:
        for name in ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
                      "WenQuanYi Micro Hei", "arial"]:
            try:
                _PIECE_FONT = pygame.font.SysFont(name, SQ_SIZE - 12)
                return
            except Exception:
                pass
        _PIECE_FONT = pygame.font.Font(None, SQ_SIZE - 12)

# ── Board data ──────────────────────────────────────────────────────────────

class BoardData:
    """Structured xiangqi board state.

    pieces: dict {(row, col): (player, piece_type)} — only occupied cells.
    cur_player: 0 (Red) or 1 (Black).
    num_distinct_actions: 8100 for xiangqi.
    """
    __slots__ = ("pieces", "cur_player")

    def __init__(self, pieces=None, cur_player=0):
        self.pieces = pieces or {}
        self.cur_player = cur_player


# ── Observation parsing ─────────────────────────────────────────────────────

def obs_to_board(obs):
    """Convert [15, 10, 9] observation to BoardData.

    Planes 0-6: Red pieces, planes 7-13: Black pieces, plane 14: player-to-move.
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(15, ROWS, COLS)
    cur_player = 0 if obs[14, 0, 0] == 1.0 else 1
    pieces = {}
    for player in (0, 1):
        base_plane = player * 7
        for ptype in range(1, 8):
            plane = base_plane + ptype - 1
            rows, cols = np.where(obs[plane] > 0.5)
            for r, c in zip(rows, cols):
                pieces[(int(r), int(c))] = (player, ptype)
    return BoardData(pieces=pieces, cur_player=cur_player)


# ── Action labels ───────────────────────────────────────────────────────────

def _sq_to_str(sq):
    """Square index 0-89 → (row, col) string like '(2,4)'."""
    row, col = sq // COLS, sq % COLS
    return f"({row},{col})"


def action_label(action):
    """Convert xiangqi action to human-readable string like '炮(7,1)→(7,4)'."""
    from_sq = action // 90
    to_sq = action % 90
    return f"{_sq_to_str(from_sq)}→{_sq_to_str(to_sq)}"


def action_label_with_piece(action, pieces):
    """Like action_label but includes piece name if known."""
    from_sq = action // 90
    to_sq = action % 90
    from_pos = (from_sq // COLS, from_sq % COLS)
    piece_info = pieces.get(from_pos)
    if piece_info:
        player, ptype = piece_info
        glyphs = _RED_PIECES if player == 0 else _BLACK_PIECES
        name = glyphs.get(ptype, "?")
    else:
        name = "?"
    return f"{name}{_sq_to_str(from_sq)}→{_sq_to_str(to_sq)}"


# ── Input ───────────────────────────────────────────────────────────────────

def click_to_square(pos):
    """Convert mouse position to (row, col), or None if outside board."""
    x, y = pos
    col = (x - LABEL_MARGIN) // SQ_SIZE
    row = y // SQ_SIZE
    if 0 <= row < ROWS and 0 <= col < COLS:
        return (row, col)
    return None


class ActionSelector:
    """Two-click action selection: first click picks source, second picks target."""

    def __init__(self):
        self._selected_src = None      # (row, col) or None

    @property
    def selected_src(self):
        return self._selected_src

    def handle_click(self, pos, state):
        """Process a click. Returns (action, src, tgt) or (None, src, None)."""
        sq = click_to_square(pos)
        if sq is None:
            self._selected_src = None
            return None, None, None

        row, col = sq
        legal = state.legal_actions()
        mask = np.asarray(state.legal_actions_mask(), dtype=bool)
        cur_player = state.current_player()

        if self._selected_src is None:
            # First click: select source piece
            src_sq = row * COLS + col
            # Check if any legal action starts from this square
            has_legal = any(
                mask[a] and (a // 90) == src_sq
                for a in range(src_sq * 90, min((src_sq + 1) * 90, 8100))
            )
            if has_legal:
                self._selected_src = (row, col)
                return None, (row, col), None
            return None, None, None

        else:
            # Second click: choose target
            src_sq = self._selected_src[0] * COLS + self._selected_src[1]
            tgt_sq = row * COLS + col
            action = src_sq * 90 + tgt_sq

            # Click same source → deselect
            if self._selected_src == (row, col):
                self._selected_src = None
                return None, None, None

            self._selected_src = None
            if action in legal:
                return action, self._selected_src, (row, col)
            # Click invalid target — try to reselect source
            return self._handle_click_retry(pos, state)

    def _handle_click_retry(self, pos, state):
        """Invalid target click: try to select the clicked square as new source."""
        sq = click_to_square(pos)
        if sq is None:
            return None, None, None
        row, col = sq
        mask = np.asarray(state.legal_actions_mask(), dtype=bool)
        src_sq = row * COLS + col
        has_legal = any(
            mask[a] and (a // 90) == src_sq
            for a in range(src_sq * 90, min((src_sq + 1) * 90, 8100))
        )
        if has_legal:
            self._selected_src = (row, col)
            return None, (row, col), None
        return None, None, None

    def reset(self):
        self._selected_src = None


# ── Board drawing ───────────────────────────────────────────────────────────

def draw_board_bg(screen):
    """10×9 grid with palace markings and river."""
    ox = LABEL_MARGIN
    board_h = ROWS * SQ_SIZE
    board_w = COLS * SQ_SIZE

    # Background
    pygame.draw.rect(screen, BG_COLOR, (ox, 0, board_w, board_h))

    # Horizontal lines — span between edge verticals
    x_left = ox + SQ_SIZE // 2
    x_right = ox + (COLS - 1) * SQ_SIZE + SQ_SIZE // 2
    y_top = SQ_SIZE // 2
    y_bot = (ROWS - 1) * SQ_SIZE + SQ_SIZE // 2

    for row in range(ROWS):
        y = row * SQ_SIZE + SQ_SIZE // 2
        pygame.draw.line(screen, LINE_COLOR, (x_left, y), (x_right, y), 1)

    # Vertical lines — split at river
    for col in range(COLS):
        x = ox + col * SQ_SIZE + SQ_SIZE // 2

        if col in (0, COLS - 1):
            pygame.draw.line(screen, LINE_COLOR, (x, y_top), (x, y_bot), 1)
        else:
            pygame.draw.line(screen, LINE_COLOR, (x, y_top),
                             (x, 4 * SQ_SIZE + SQ_SIZE // 2), 1)
            pygame.draw.line(screen, LINE_COLOR,
                             (x, 5 * SQ_SIZE + SQ_SIZE // 2), (x, y_bot), 1)

    # Palace diagonals (top: rows 0-2 cols 3-5, bottom: rows 7-9 cols 3-5)
    for (r0, c0), (r1, c1) in [((0, 3), (2, 5)), ((0, 5), (2, 3)),
                                 ((7, 3), (9, 5)), ((7, 5), (9, 3))]:
        x0 = ox + c0 * SQ_SIZE + SQ_SIZE // 2
        y0 = r0 * SQ_SIZE + SQ_SIZE // 2
        x1 = ox + c1 * SQ_SIZE + SQ_SIZE // 2
        y1 = r1 * SQ_SIZE + SQ_SIZE // 2
        pygame.draw.line(screen, LINE_COLOR, (x0, y0), (x1, y1), 1)

    # River text
    f = get_font(24)
    river_y = 4 * SQ_SIZE + SQ_SIZE
    river_label = f.render("楚河          漢界", True, LINE_COLOR)
    screen.blit(river_label,
                (ox + board_w // 2 - river_label.get_width() // 2,
                 river_y - river_label.get_height() // 2))

    # Row/col labels
    f_sm = get_font(16)
    for i in range(ROWS):
        t = f_sm.render(str(i), True, GRAY)
        screen.blit(t, (2, i * SQ_SIZE + SQ_SIZE // 2 - t.get_height() // 2))
    for i in range(COLS):
        t = f_sm.render(str(i), True, GRAY)
        screen.blit(t, (ox + i * SQ_SIZE + SQ_SIZE // 2 - t.get_width() // 2,
                        board_h + 4))


def draw_pieces(screen, pieces, highlight_src=None):
    """Draw Chinese character pieces. *highlight_src*: (row, col) to highlight."""
    _ensure_piece_font()
    font = _PIECE_FONT
    ox = LABEL_MARGIN

    for (row, col), (player, ptype) in pieces.items():
        glyphs = _RED_PIECES if player == 0 else _BLACK_PIECES
        glyph = glyphs.get(ptype, "?")
        color = _PIECE_COLORS[player]

        cx = ox + col * SQ_SIZE + SQ_SIZE // 2
        cy = row * SQ_SIZE + SQ_SIZE // 2

        # Highlight ring
        if highlight_src == (row, col):
            pygame.draw.circle(screen, YELLOW, (cx, cy), SQ_SIZE // 2 - 2, 3)

        # Piece circle background
        radius = SQ_SIZE // 2 - 4
        pygame.draw.circle(screen, BG_COLOR, (cx, cy), radius)
        pygame.draw.circle(screen, LINE_COLOR, (cx, cy), radius, 2)

        # Glyph
        t = font.render(glyph, True, color)
        screen.blit(t, (cx - t.get_width() // 2, cy - t.get_height() // 2))


def draw_legal_dots(screen, legal_targets):
    """Small dots indicating legal target squares for the selected source."""
    ox = LABEL_MARGIN
    for row, col in legal_targets:
        cx = ox + col * SQ_SIZE + SQ_SIZE // 2
        cy = row * SQ_SIZE + SQ_SIZE // 2
        pygame.draw.circle(screen, GREEN, (cx, cy), 6)


def hint_arrows(actions, visits, top_n=5):
    """Convert MCTS hint data to arrow list. Returns [(from_sq, to_sq, weight), ...]."""
    if not actions:
        return []
    total = sum(visits) or 1
    moves = [(a // 90, a % 90, float(v / total)) for a, v in zip(actions, visits)]
    moves.sort(key=lambda x: -x[2])
    return moves[:top_n]


def compute_top_moves(policy, mask, top_n=5):
    """Extract top-N legal moves from a flat policy array.

    Returns list of (from_sq, to_sq, weight) sorted by probability.
    """
    if policy is None:
        return []
    moves = []
    for a in range(len(policy)):
        if mask is not None and not mask[a]:
            continue
        p = policy[a]
        if p < 0.001:
            continue
        from_sq = a // 90
        to_sq = a % 90
        moves.append((from_sq, to_sq, float(p)))
    moves.sort(key=lambda x: -x[2])
    return moves[:top_n]


def draw_move_arrows(screen, moves):
    """Draw arrows for top policy moves.

    *moves*: list of (from_sq, to_sq, weight) where weight is 0.0–1.0.
    """
    if not moves:
        return
    ox = LABEL_MARGIN
    max_w = max(w for _, _, w in moves) or 1.0

    for from_sq, to_sq, weight in moves:
        sr, sc = from_sq // COLS, from_sq % COLS
        tr, tc = to_sq // COLS, to_sq % COLS
        # Skip same-square (shouldn't happen for legal moves)
        if (sr, sc) == (tr, tc):
            continue

        x0 = ox + sc * SQ_SIZE + SQ_SIZE // 2
        y0 = sr * SQ_SIZE + SQ_SIZE // 2
        x1 = ox + tc * SQ_SIZE + SQ_SIZE // 2
        y1 = tr * SQ_SIZE + SQ_SIZE // 2

        # Offset from source center toward target so arrow doesn't cover the piece
        dx, dy = x1 - x0, y1 - y0
        dist = max((dx**2 + dy**2)**0.5, 1.0)
        ux, uy = dx / dist, dy / dist
        offset = SQ_SIZE // 2 - 2
        x0 += ux * offset
        y0 += uy * offset

        # Arrowhead lands before target piece center
        x1 -= ux * offset
        y1 -= uy * offset

        # Color: red=high weight, blue=low; alpha from weight
        alpha = int(80 + 160 * (weight / max_w))
        r = int(220 * (weight / max_w))
        g = 40
        b = int(220 * (1.0 - weight / max_w))
        color = (r, g, b, alpha)

        # Draw on a temporary surface for alpha blending
        arrow_surf = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
        pygame.draw.line(arrow_surf, color, (x0, y0), (x1, y1), max(2, int(5 * weight / max_w)))

        # Arrowhead
        arrow_len = 10
        angle = 0.5
        ax1 = x1 - ux * arrow_len + uy * arrow_len * angle
        ay1 = y1 - uy * arrow_len - ux * arrow_len * angle
        ax2 = x1 - ux * arrow_len - uy * arrow_len * angle
        ay2 = y1 - uy * arrow_len + ux * arrow_len * angle
        pygame.draw.polygon(arrow_surf, color, [(x1, y1), (ax1, ay1), (ax2, ay2)])

        screen.blit(arrow_surf, (0, 0))


def draw_source_heatmap(screen, src_probs):
    """Heatmap on source squares — which pieces are likely to be moved.

    *src_probs*: dict {sq: probability} where sq = row*9+col.
    """
    ox = LABEL_MARGIN
    for sq, prob in src_probs.items():
        row, col = sq // COLS, sq % COLS
        intensity = min(prob * 3, 1.0)
        s = heatmap_surface(SQ_SIZE, intensity)
        screen.blit(s, (ox + col * SQ_SIZE, row * SQ_SIZE))


def draw_target_heatmap(screen, src_sq, policy):
    """Heatmap on target squares for a selected source square.

    *src_sq*: source square index (0-89).
    *policy*: flat policy array (8100,).
    """
    ox = LABEL_MARGIN
    start = src_sq * 90
    end = min(start + 90, len(policy))
    for tgt_sq in range(90):
        idx = start + tgt_sq
        if idx >= end:
            break
        p = policy[idx]
        if p > 0.001:
            row, col = tgt_sq // COLS, tgt_sq % COLS
            s = heatmap_surface(SQ_SIZE, min(p * 5, 1.0))
            screen.blit(s, (ox + col * SQ_SIZE, row * SQ_SIZE))


def compute_source_probs(policy):
    """Aggregate policy: sum probs for each source square.  Returns dict {sq: prob}."""
    src_probs = {}
    for from_sq in range(90):
        start = from_sq * 90
        end = start + 90
        p = policy[start:end].sum()
        if p > 0.001:
            src_probs[from_sq] = float(p)
    return src_probs


# ── MCTS hints ──────────────────────────────────────────────────────────────

def draw_mcts_hints(screen, actions, visits, q_values):
    """Overlay MCTS visit counts on source squares."""
    if not actions or not visits or not q_values:
        return
    ox = LABEL_MARGIN
    max_visit = max(visits)
    best_q = max(q_values)
    font_v = get_font(20)
    font_q = get_font(20)

    for a, vst, q in zip(actions, visits, q_values):
        from_sq = a // 90
        row, col = from_sq // COLS, from_sq % COLS
        cx = ox + col * SQ_SIZE + SQ_SIZE // 2
        cy = row * SQ_SIZE + SQ_SIZE // 2

        v_color = RED if (vst == max_visit and max_visit > 1) else (
            BLUE if vst > max_visit * 0.3 else GRAY)
        v_txt = font_v.render(str(vst), True, v_color)
        screen.blit(v_txt, (cx - v_txt.get_width() // 2, cy + SQ_SIZE // 2 - 20))

        q_color = GRAY if q == best_q else BLACK
        q_txt = font_q.render(f"{q:+.2f}", True, q_color)
        screen.blit(q_txt, (cx - q_txt.get_width() // 2, cy - SQ_SIZE // 2 + 4))


# ── Side panel (for buffer viewer) ──────────────────────────────────────────

def build_display_data(board_data, policy, legal_mask, value, tag, cur_player):
    """Pre-compute display data for one buffer sample."""
    d = {}
    d["pieces"] = board_data.pieces
    d["cur_player"] = cur_player
    d["src_probs"] = compute_source_probs(policy) if policy is not None else {}

    v = np.asarray(value, dtype=np.float32).ravel()
    if len(v) == 3:
        d["wdl"] = (float(v[0]), float(v[1]), float(v[2]))
        d["value"] = float(v[0] - v[2])
    else:
        d["wdl"] = None
        d["value"] = float(v[0])
    d["win_pct"] = (1 + d["value"]) * 50

    d["tag_label"] = tag_display(tag)
    d["tag_color"] = TAG_COLORS.get(d["tag_label"], BLACK)

    # Top-5 policy
    if policy is not None:
        top5 = sorted([(a, policy[a]) for a in range(len(policy))],
                      key=lambda x: -x[1])[:5]
        d["top5"] = [(a, p, legal_mask is not None and legal_mask[a])
                      for a, p in top5]
    else:
        d["top5"] = []

    return d


def draw_board_viewer(screen, disp, show_heatmap=True):
    """Combined xiangqi board render for buffer viewer."""
    draw_board_bg(screen)
    if show_heatmap:
        draw_source_heatmap(screen, disp["src_probs"])
    draw_pieces(screen, disp["pieces"])


# ── UI draw helper ──────────────────────────────────────────────────────────

def draw_board_ui(screen, board_data, legal_actions=None, selected_src=None,
                  hint_data=None, hint_heatmap=None, move_arrows=None):
    """Combined xiangqi board render for game UI.

    *selected_src*: (row, col) of selected source piece.
    *hint_data*: (actions, visits, q_values) from MCTS hints.
    *hint_heatmap*: dict {sq: prob} source-aggregated hint probs.
    *move_arrows*: list of (from_sq, to_sq, weight) for arrow overlay.
    """
    draw_board_bg(screen)

    if move_arrows:
        draw_move_arrows(screen, move_arrows)

    if hint_heatmap:
        draw_source_heatmap(screen, hint_heatmap)
    elif hint_data is not None:
        pass  # draw_mcts_hints called separately

    draw_pieces(screen, board_data.pieces, highlight_src=selected_src)

    if legal_actions is not None and selected_src is not None:
        src_sq = selected_src[0] * COLS + selected_src[1]
        targets = set()
        for a in legal_actions:
            if a // 90 == src_sq:
                tgt_sq = a % 90
                targets.add((tgt_sq // COLS, tgt_sq % COLS))
        draw_legal_dots(screen, list(targets))

    if hint_data is not None:
        actions, visits, q_values = hint_data
        draw_mcts_hints(screen, actions, visits, q_values)
