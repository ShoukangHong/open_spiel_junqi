"""Visual browser for Othello replay buffer (.npz) files.

Usage:
    python buffer_viewer.py

Controls:
    Left/Right  — ±1 sample           Up/Down   — ±10 samples
    PgUp/PgDn   — ±100 samples        Home/End  — first / last
    H           — toggle policy heatmap
    T           — cycle tag filter   R — filter rare only   A — show all
    Q / Esc     — quit
"""

import os
import numpy as np
import pygame

# ── Config ──────────────────────────────────────────────────────────────────────
BUFFER_FILE = r"C:\Users\shouk\othello_train_cloud\buffer-checkpoint-140.npz"

# ── Constants ───────────────────────────────────────────────────────────────────
ROWS = COLS = 8
SQ_SIZE = 74
LABEL_MARGIN = 20
BOARD_W = LABEL_MARGIN + COLS * SQ_SIZE                    # 612
BOARD_H = ROWS * SQ_SIZE + 18                               # 610
PANEL_W = 290
PADDING = 10
WIDTH = BOARD_W + PANEL_W                                   # 902
HEIGHT = BOARD_H                                            # 610

# Colors
GREEN = (0, 128, 0)
DARK_GREEN = (0, 100, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
BG = (240, 240, 240)

TAG_COLORS = {
    "": BLACK, "normal": BLACK,
    "rare": RED, "weak": ORANGE, "weak_final": (200, 100, 0),
}
TAG_FILTERS = ["all", "normal", "rare", "weak"]

# ── Cached fonts (created once) ─────────────────────────────────────────────────
_FONT_LABEL = None       # 18
_FONT_POLICY_BIG = None  # 24
_FONT_POLICY_MID = None  # 20
_FONT_PANEL = None       # 22
_FONT_PANEL_SM = None    # 16

def _init_fonts():
    global _FONT_LABEL, _FONT_POLICY_BIG, _FONT_POLICY_MID, _FONT_PANEL, _FONT_PANEL_SM
    if _FONT_LABEL is None:
        _FONT_LABEL = pygame.font.Font(None, 18)
        _FONT_POLICY_BIG = pygame.font.Font(None, 24)
        _FONT_POLICY_MID = pygame.font.Font(None, 20)
        _FONT_PANEL = pygame.font.Font(None, 22)
        _FONT_PANEL_SM = pygame.font.Font(None, 16)

# ── Pre-rendered heatmap tiles (created once per SQ_SIZE) ───────────────────────
_HEATMAP_CACHE = {}  # intensity_int → Surface

def _heatmap_surface(intensity):
    """Return a cached transparent Surface for the given intensity [0.0, 1.0]."""
    key = int(intensity * 100)  # 0..100
    if key not in _HEATMAP_CACHE:
        c = (int(255 * intensity), int(128 * (1 - intensity)),
             int(128 * (1 - intensity)))
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill((c[0], c[1], c[2], 120))
        _HEATMAP_CACHE[key] = s
    return _HEATMAP_CACHE[key]

# ── Data helpers ────────────────────────────────────────────────────────────────

def obs_to_board(obs):
    obs = np.asarray(obs, dtype=np.float32).reshape(4, 8, 8)
    cur = 0 if obs[3, 0, 0] == 1 else 1
    return (obs[1] - obs[2]).flatten().astype(int), cur


def _action_label(a):
    if a >= 64:
        return "pass"
    return f"{chr(ord('a') + a % 8)}{a // 8 + 1}"


def load_buffer(path):
    data = np.load(path, allow_pickle=True)
    tags = None
    if "tags" in data:
        tags = np.asarray(data["tags"], dtype=str)
    return data["obs"], data["masks"], data["policies"], data["values"], tags


def build_filtered_indices(tags, total, tag_filter):
    if tag_filter == "all" or tags is None:
        return np.arange(total, dtype=int)
    if tag_filter == "normal":
        mask = np.asarray([t in ("", "normal", "None") for t in tags])
    elif tag_filter == "weak":
        mask = np.asarray([t in ("weak", "weak_final") for t in tags])
    else:
        mask = np.asarray(tags == tag_filter)
    idx = np.where(mask)[0]
    return idx if len(idx) > 0 else np.array([], dtype=int)


def _tag_display(tag):
    if tag is None or tag == "" or tag == "None":
        return "normal"
    return str(tag)


# ── Pre-computed display data (updated only when sample changes) ───────────────

def build_display_data(board, policy, legal_mask, value, tag, cur_player):
    """Compute everything needed for rendering. Called ONCE per sample switch."""
    d = {}

    # Board basics
    d["board"] = board
    d["cur_player"] = cur_player
    d["black_c"] = int((board == 1).sum())
    d["white_c"] = int((board == -1).sum())

    # Value
    d["value"] = value
    d["win_pct"] = (1 + value) * 50

    # Tag
    d["tag_label"] = _tag_display(tag)
    d["tag_color"] = TAG_COLORS.get(d["tag_label"], BLACK)

    # Policy hints (pre-compute positions and colors)
    hints = []
    if policy is not None and legal_mask is not None:
        legal_actions = [a for a in range(65) if legal_mask[a]]
        if legal_actions:
            best_action = max(legal_actions, key=lambda a: policy[a])
        else:
            best_action = -1
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

    # Top-5 policy (sorted)
    if policy is not None:
        top5 = sorted([(a, policy[a]) for a in range(65)], key=lambda x: -x[1])[:5]
        d["top5"] = [(a, p, legal_mask is not None and legal_mask[a])
                      for a, p in top5]
    else:
        d["top5"] = []

    # Heatmap intensities
    heatmap_cells = []
    if policy is not None:
        for a in range(64):
            p = policy[a]
            if p > 0.001:
                heatmap_cells.append((a, min(p * 3, 1.0)))
    d["heatmap_cells"] = heatmap_cells

    return d


# ── Drawing (uses pre-computed data, no per-frame allocations) ──────────────────

def draw_board(screen, disp, show_heatmap):
    ox = LABEL_MARGIN
    board_h = ROWS * SQ_SIZE

    # Board background
    pygame.draw.rect(screen, GREEN, (ox, 0, COLS * SQ_SIZE, board_h))

    # Grid
    for i in range(COLS + 1):
        x = ox + i * SQ_SIZE
        pygame.draw.line(screen, BLACK, (x, 0), (x, board_h), 2)
        pygame.draw.line(screen, BLACK, (ox, i * SQ_SIZE),
                         (ox + COLS * SQ_SIZE, i * SQ_SIZE), 2)

    # Labels
    f = _FONT_LABEL
    for i in range(ROWS):
        t = f.render(str(i + 1), True, BLACK)
        screen.blit(t, (2, i * SQ_SIZE + SQ_SIZE // 2 - t.get_height() // 2))
    for i in range(COLS):
        t = f.render(chr(ord('a') + i), True, BLACK)
        screen.blit(t, (ox + i * SQ_SIZE + SQ_SIZE // 2 - t.get_width() // 2,
                        board_h + 2))

    # Heatmap — uses cached surfaces
    if show_heatmap:
        for a, intensity in disp["heatmap_cells"]:
            row, col = a // COLS, a % COLS
            screen.blit(_heatmap_surface(intensity),
                        (ox + col * SQ_SIZE, row * SQ_SIZE))

    # Pieces
    board = disp["board"]
    for row in range(ROWS):
        for col in range(COLS):
            cx = ox + col * SQ_SIZE + SQ_SIZE // 2
            cy = row * SQ_SIZE + SQ_SIZE // 2
            v = board[row * COLS + col]
            if v == 1:
                pygame.draw.circle(screen, BLACK, (cx, cy), SQ_SIZE // 2 - 4)
            elif v == -1:
                pygame.draw.circle(screen, WHITE, (cx, cy), SQ_SIZE // 2 - 4)

    # Policy hints (pre-computed, just blit)
    fbig = _FONT_POLICY_BIG
    fmid = _FONT_POLICY_MID
    for a, p, clr, size in disp["hints"]:
        row, col = a // COLS, a % COLS
        cx = ox + col * SQ_SIZE + SQ_SIZE // 2
        cy = row * SQ_SIZE + SQ_SIZE // 2
        font = fbig if size == "big" else fmid
        t = font.render(f"{p:.2f}", True, clr)
        screen.blit(t, (cx - t.get_width() // 2,
                        cy + SQ_SIZE // 2 - t.get_height() - 2))


def draw_side_panel(screen, index, total, global_idx, tag_filter,
                    disp, filtered_indices, tags_arr):
    x0 = BOARD_W + PADDING
    y = PADDING
    w = PANEL_W - 2 * PADDING
    f = _FONT_PANEL
    sf = _FONT_PANEL_SM

    def _line(text, color=BLACK, font=f, advance=True):
        nonlocal y
        s = font.render(text, True, color)
        screen.blit(s, (x0, y))
        if advance:
            y += font.get_height() + 3

    def _hline():
        nonlocal y
        pygame.draw.line(screen, LIGHT_GRAY, (x0, y), (x0 + w, y))
        y += 6

    # ── Sample info ─────────────────────────────────────────────────────────
    _line(f"Sample #{global_idx}  ({index + 1}/{total})", font=f)
    pname = ("BLACK" if disp["cur_player"] == 0
             else ("WHITE" if disp["cur_player"] == 1 else "?"))
    _line(f"Turn: {pname}    B:{disp['black_c']}  W:{disp['white_c']}")
    _line(f"Value: {disp['value']:+.3f}  ({disp['win_pct']:.1f}%)")

    _line("Tag: ", advance=False)
    x_after = x0 + f.render("Tag: ", True, BLACK).get_width()
    s = f.render(disp["tag_label"], True, disp["tag_color"])
    screen.blit(s, (x_after, y))
    y += f.get_height() + 3

    _line(f"Filter: [{tag_filter}]", color=GRAY)
    _hline()

    # ── Top-5 policy ─────────────────────────────────────────────────────────
    _line("Policy:", font=f)
    bar_max = 90
    bx0 = x0 + 68
    for a, p, legal in disp["top5"]:
        label = _action_label(a)
        clr = BLACK if legal else GRAY
        screen.blit(sf.render(f"  {label:>5s}", True, clr), (x0, y))
        bw = int(p * bar_max)
        bar_clr = (200, 50, 50) if p > 0.1 else (140, 140, 140)
        if bw > 0:
            pygame.draw.rect(screen, bar_clr, (bx0, y + 2, bw, 10))
        screen.blit(sf.render(f" {p:.3f}", True, BLACK), (bx0 + bw + 3, y))
        y += sf.get_height() + 2
    _hline()

    # ── Browse ───────────────────────────────────────────────────────────────
    _line("Browse:", font=f)
    n_show = 12
    half = n_show // 2
    total_f = len(filtered_indices)
    if total_f == 0:
        _line("  (no entries)", color=GRAY, font=sf)
    else:
        start = max(0, min(index - half, total_f - n_show))
        end = min(total_f, start + n_show)
        for gi in filtered_indices[start:end]:
            gi_int = int(gi)
            is_current = (gi_int == global_idx)
            marker = "▶" if is_current else " "
            label = f"{marker} {gi_int:>5d}"
            t = str(tags_arr[gi_int]) if tags_arr is not None else ""
            tag_disp = _tag_display(t)
            tag_clr = TAG_COLORS.get(tag_disp, BLACK)

            clr = WHITE if is_current else BLACK
            bg_clr = (100, 140, 200) if is_current else None
            if bg_clr is not None:
                pygame.draw.rect(screen, bg_clr, (x0, y, w, sf.get_height() + 2))
            screen.blit(sf.render(label, True, clr), (x0 + 2, y + 1))
            if tag_disp != "normal":
                tw = sf.render(label, True, clr).get_width()
                screen.blit(sf.render(f"[{tag_disp}]", True, tag_clr),
                            (x0 + tw + 10, y + 1))
            y += sf.get_height() + 2
    _hline()

    # ── Key hints ────────────────────────────────────────────────────────────
    for line in [
        "← → : ±1     ↑↓ : ±10",
        "PgUp/Dn: ±100   Home/End",
        "H: heatmap   T: cycle filter",
        "R: rare only   A: show all",
        "Q/Esc: quit",
    ]:
        screen.blit(sf.render(line, True, GRAY), (x0, y))
        y += sf.get_height() + 1


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    path = BUFFER_FILE
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
    _init_fonts()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Buffer Viewer — {os.path.basename(path)}")
    clock = pygame.time.Clock()

    # Mutable state
    _s = {"filter": "all", "filtered_idx": None, "pos": 0, "heatmap": True}
    _s["filtered_idx"] = build_filtered_indices(tags_arr, total_global, "all")

    def _do_filter(name):
        _s["filter"] = name
        _s["filtered_idx"] = build_filtered_indices(tags_arr, total_global, name)
        _s["pos"] = 0
        print(f"[filter] → {name}  ({len(_s['filtered_idx'])} entries)")

    # Pre-computed display data — updated on sample switch or filter change
    _disp = None
    _last_gi = -1

    def _refresh_disp():
        nonlocal _disp, _last_gi
        f = _s["filtered_idx"]
        if len(f) == 0:
            _disp = None
            return
        pos = _s["pos"] % len(f)
        g_idx = int(f[pos])
        tag = tags_arr[g_idx] if tags_arr is not None else None
        board, cp = obs_to_board(obs_arr[g_idx])
        _disp = build_display_data(
            board, policies_arr[g_idx], masks_arr[g_idx],
            values_arr[g_idx], tag, cp)
        _last_gi = g_idx

    _refresh_disp()

    running = True
    while running:
        f = _s["filtered_idx"]
        if len(f) == 0:
            # Nothing to show — just process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        _do_filter("all")
                    elif event.key == pygame.K_t:
                        idx = (TAG_FILTERS.index(_s["filter"]) + 1) % len(TAG_FILTERS)
                        _do_filter(TAG_FILTERS[idx])
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
            if _s["filtered_idx"] is not None and len(_s["filtered_idx"]) > 0:
                _refresh_disp()
            screen.fill(BG)
            pygame.display.flip()
            clock.tick(30)
            continue

        pos = _s["pos"] % len(f)
        g_idx = int(f[pos])

        # ── Process events ──────────────────────────────────────────────────
        need_refresh = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                n = len(f)
                key = event.key
                if key == pygame.K_RIGHT:
                    _s["pos"] = (_s["pos"] + 1) % n
                    need_refresh = True
                elif key == pygame.K_LEFT:
                    _s["pos"] = (_s["pos"] - 1) % n
                    need_refresh = True
                elif key == pygame.K_UP:
                    _s["pos"] = (_s["pos"] + 10) % n
                    need_refresh = True
                elif key == pygame.K_DOWN:
                    _s["pos"] = (_s["pos"] - 10) % n
                    need_refresh = True
                elif key == pygame.K_PAGEUP:
                    _s["pos"] = (_s["pos"] + 100) % n
                    need_refresh = True
                elif key == pygame.K_PAGEDOWN:
                    _s["pos"] = (_s["pos"] - 100) % n
                    need_refresh = True
                elif key == pygame.K_HOME:
                    _s["pos"] = 0
                    need_refresh = True
                elif key == pygame.K_END:
                    _s["pos"] = n - 1
                    need_refresh = True
                elif key == pygame.K_h:
                    _s["heatmap"] = not _s["heatmap"]
                elif key == pygame.K_r:
                    _do_filter("rare")
                    need_refresh = True
                elif key == pygame.K_t:
                    idx = (TAG_FILTERS.index(_s["filter"]) + 1) % len(TAG_FILTERS)
                    _do_filter(TAG_FILTERS[idx])
                    need_refresh = True
                elif key == pygame.K_a:
                    _do_filter("all")
                    need_refresh = True
                elif key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        # ── Re-sync after filter change ─────────────────────────────────────
        if need_refresh:
            f = _s["filtered_idx"]
            if len(f) > 0:
                _refresh_disp()
            else:
                _disp = None

        # ── Render ──────────────────────────────────────────────────────────
        screen.fill(BG)
        if _disp is not None and len(f) > 0:
            pos = _s["pos"] % len(f)
            g_idx = int(f[pos])
            draw_board(screen, _disp, _s["heatmap"])
            draw_side_panel(screen, pos, len(f), g_idx, _s["filter"],
                            _disp, f, tags_arr)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
