"""Shared UI engine — game-agnostic utilities for game_ui and buffer_viewer.

Provides:
  - Model loading / bot creation
  - MCTSHintEngine (persistent search + freeze + subtree inheritance)
  - UI widgets (color menu, game over screen)
  - Buffer loading (SQLite/npz)
  - TagFilter + navigation state
  - Console MCTS info printer
"""

import json
import os
import sqlite3
import zlib
import sys

import numpy as np
import pygame
import pyspiel

from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.evaluator import PyTorchEvaluator
from train.batch_mcts.mcts import BatchMCTS

# ── Tag constants (shared between viewer and UI) ────────────────────────────

TAG_COLORS = {
    "": (0, 0, 0), "normal": (0, 0, 0),
    "rare": (255, 0, 0), "rare_flip": (128, 0, 128),
    "weak": (255, 165, 0), "weak_final": (200, 100, 0),
}
TAG_FILTERS = ["all", "normal", "rare", "rare_flip", "weak"]

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 128, 0)
BG = (240, 240, 240)

# ── Model / bot utilities ───────────────────────────────────────────────────

def load_model_for_ui(checkpoint_dir, checkpoint_step, build_model_fn):
    """Load a PyTorch model from a training checkpoint directory."""
    config_path = os.path.join(checkpoint_dir, "train_config.json")
    with open(config_path) as f:
        train_cfg = json.load(f)
    train_cfg["path"] = checkpoint_dir
    game = pyspiel.load_game(train_cfg["game"])
    model = build_model_fn(game, train_cfg)
    model.load_checkpoint(checkpoint_step)
    print(f"Loaded checkpoint-{checkpoint_step} from {checkpoint_dir}")
    print(f"  params={model.num_trainable_variables}")
    return model, game


def create_bot(game, model, mcts_sims, batch_size, uct_c,
               random_state=None):
    """Create a BatchMCTS bot backed by a PyTorch model."""
    evaluator = PyTorchEvaluator(game, model)
    mcts_cfg = MCTSConfig(
        max_simulations=mcts_sims, batch_size=batch_size,
        uct_c=uct_c, policy_epsilon=0, verbose=False)
    bot = BatchMCTS(game, mcts_cfg, evaluator,
                    random_state=random_state or np.random.RandomState())
    return bot, evaluator, mcts_cfg


# ── MCTS Hint Engine ────────────────────────────────────────────────────────

class MCTSHintEngine:
    """Persistent MCTS search for interactive hints in game UI.

    - Accumulates simulations across frames (incremental search).
    - Supports freeze (lock current state to study).
    - Supports subtree inheritance (reuse child node after a move).
    """

    def __init__(self, game, evaluator, max_sims=12800, batch_size=16,
                 uct_c=1.41):
        self._game = game
        self._evaluator = evaluator
        self._max_sims = max_sims
        self._cfg = MCTSConfig(max_simulations=64, batch_size=batch_size,
                               uct_c=uct_c, policy_epsilon=0, verbose=False)
        self._mcts = BatchMCTS(game, self._cfg, evaluator,
                              random_state=np.random.RandomState())
        self._root = None
        self._state_key = None
        self._frozen = False
        self._frozen_root = None
        self._frozen_key = None

    def search(self, state):
        """Run incremental search, returning (actions, visits, q_values, root_value, draw_rate)."""
        key = str(state)

        if self._frozen:
            if self._frozen_root is not None and self._frozen_key == key:
                return self._extract_hints(self._frozen_root)
            return self._extract_hints(self._frozen_root)

        # Reset if state changed
        if self._root is None or self._state_key != key:
            self._cfg.max_simulations = 64
            self._mcts = BatchMCTS(self._game, self._cfg, self._evaluator,
                                  random_state=np.random.RandomState())
            self._root = self._mcts.mcts_search(state)
            self._state_key = key
        elif self._root.visit_count < self._max_sims:
            self._cfg.max_simulations = 64
            self._root = self._mcts.mcts_search(state, root=self._root)

        return self._extract_hints(self._root)

    def freeze(self):
        """Freeze current search state."""
        self._frozen = True
        self._frozen_root = self._root
        self._frozen_key = self._state_key

    def unfreeze(self):
        self._frozen = False

    @property
    def is_frozen(self):
        return self._frozen

    def subtree_inherit(self, action):
        """After a move, try to reuse the child subtree."""
        if self._root is None:
            return
        for c in self._root.children:
            if c.action == action:
                self._root = c
                return
        self._root = None

    def _extract_hints(self, root):
        if root is None:
            return [], [], [], 0.0, 0.0
        actions = [c.action for c in root.children]
        visits = [c.explore_count for c in root.children]
        q_values = []
        for c in root.children:
            if c.outcome is not None:
                q_values.append(c.outcome[c.player])
            else:
                q_values.append(c.q_value)
        if root.outcome is not None:
            root_val = root.outcome[root.player]
        else:
            root_val = root.total_reward / max(root.explore_count, 1)
        return actions, visits, q_values, root_val, root.draw_rate


# ── UI widgets ──────────────────────────────────────────────────────────────

def choose_color_menu(screen, width, height):
    """Menu: pick black/white/HvH. Returns 0, 1, or -1 for HvH."""
    font = pygame.font.Font(None, 40)
    btn_black = pygame.Rect(150, 200, 300, 60)
    btn_white = pygame.Rect(150, 300, 300, 60)
    btn_hvh = pygame.Rect(150, 400, 300, 60)

    while True:
        screen.fill(GREEN)
        title = font.render("Play as:", True, BLACK)
        screen.blit(title, (250, 110))

        for btn, label in [(btn_black, "Black (first)"), (btn_white, "White (second)"),
                           (btn_hvh, "Human vs Human")]:
            pygame.draw.rect(screen, GRAY, btn)
            screen.blit(font.render(label, True, WHITE), (btn.x + 50, btn.y + 14))

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


def game_over_screen(screen, width, height, result_text, board_draw_fn,
                     board_data):
    """Show game result. Returns True to restart, False to quit."""
    font = pygame.font.Font(None, 50)
    small = pygame.font.Font(None, 30)

    while True:
        screen.fill(GREEN)
        board_draw_fn(screen, board_data)

        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        txt = font.render(result_text, True, YELLOW)
        screen.blit(txt, (width // 2 - txt.get_width() // 2, height // 2 - 40))

        r_txt = small.render("R - restart   Q - quit", True, WHITE)
        screen.blit(r_txt, (width // 2 - r_txt.get_width() // 2, height // 2 + 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True
                if event.key == pygame.K_q:
                    return False
        pygame.display.flip()


# ── Console MCTS info ───────────────────────────────────────────────────────

def print_mcts_info(root, state, evaluator=None, action_label_fn=None):
    """Print MCTS root policy, NN prior, and value to console."""
    player = state.current_player()
    pname = "RED(p0)" if player == 0 else "BLACK(p1)"
    print(f"  -- MCTS  {pname} --")

    if root.outcome is not None:
        root_val = root.outcome[player]
    else:
        root_val = root.total_reward / max(root.explore_count, 1)
    draw_info = f"  draw={root.draw_rate:.3f}" if root.draw_rate > 0.001 else ""
    print(f"  value = {root_val:+.4f}{draw_info}    sims = {root.explore_count}"
          f"{'  (solved)' if root.outcome is not None else ''}")

    # NN raw
    if evaluator is not None:
        nn_value, nn_policy = evaluator._inference(state)
        if hasattr(nn_value, '__len__') and not isinstance(nn_value, float):
            w, d, l = float(nn_value[0]), float(nn_value[1]), float(nn_value[2])
            nn_val = (w - l)
            nn_str = f"w={w:.3f} d={d:.3f} l={l:.3f}"
        else:
            nn_val = float(nn_value)
            nn_str = f"{nn_val:+.4f}"
        print(f"  -- NN raw {nn_str}    "
              f"MCTS value={root_val:+.4f}{draw_info}    "
              f"sims={root.explore_count} --")

    # Top moves
    legal = state.legal_actions()
    rows = []
    for a in legal:
        mcts_n = 0
        mcts_v = 0.0
        mcts_solved = " "
        for c in root.children:
            if c.action == a:
                mcts_n = c.explore_count
                mcts_v = c.outcome[player] if c.outcome is not None else c.q_value
                mcts_solved = "✓" if c.outcome is not None else " "
                break
        nn_p = float(nn_policy[a]) if nn_policy is not None else 0.0
        rows.append((a, mcts_n, mcts_v, mcts_solved, nn_p))

    rows.sort(key=lambda r: (-r[1], -r[4]))
    total_visits = sum(c.explore_count for c in root.children)

    BAR_W = 30
    for a, mcts_n, mcts_v, mcts_solved, nn_p in rows[:12]:
        mcts_p = mcts_n / max(total_visits, 1)
        nn_bar = "█" * int(nn_p * BAR_W) if nn_p > 0.001 else ""
        mcts_bar = "█" * int(mcts_p * BAR_W) if mcts_n > 0 else ""
        label = action_label_fn(a) if action_label_fn else str(a)
        print(f"  {label:>10s}  NN={nn_p:.3f} {nn_bar:<{BAR_W}s}  "
              f"MCTS={mcts_p:.3f} {mcts_bar:<{BAR_W}s} V={mcts_v:+.3f}{mcts_solved} N={mcts_n:>4d}")
    print()


# ── Buffer loading ──────────────────────────────────────────────────────────

def _unpack_blob(raw, dtype):
    try:
        return np.frombuffer(zlib.decompress(raw), dtype=dtype)
    except zlib.error:
        return np.frombuffer(raw, dtype=dtype)


def load_buffer(path):
    """Load replay buffer from .npz or .db.  Returns (obs, masks, policies, values, tags)."""
    if path.endswith(".db"):
        return _load_buffer_db(path)
    return _load_buffer_npz(path)


def _load_buffer_npz(path):
    data = np.load(path, allow_pickle=True)
    tags = None
    if "tags" in data:
        tags = np.asarray(data["tags"], dtype=str)
    return data["obs"], data["masks"], data["policies"], data["values"], tags


def _load_buffer_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT obs, mask, policy, value, tag FROM states ORDER BY id")
    rows = cur.fetchall()
    n = len(rows)
    if n == 0:
        conn.close()
        return None, None, None, None, None
    first_obs = _unpack_blob(rows[0][0], np.float32)
    first_mask = _unpack_blob(rows[0][1], bool)
    first_pol = _unpack_blob(rows[0][2], np.float32)
    obs_arr = np.empty((n, *first_obs.shape), dtype=np.float32)
    mask_arr = np.empty((n, *first_mask.shape), dtype=bool)
    pol_arr = np.empty((n, *first_pol.shape), dtype=np.float32)
    val_arr = np.empty((n, 3), dtype=np.float32)
    tag_arr = np.empty((n,), dtype=object)
    for i, (obs_b, mask_b, pol_b, val_b, tag) in enumerate(rows):
        obs_arr[i] = _unpack_blob(obs_b, np.float32)
        mask_arr[i] = _unpack_blob(mask_b, bool)
        pol_arr[i] = _unpack_blob(pol_b, np.float32)
        val_arr[i] = _unpack_blob(val_b, np.float32)
        tag_arr[i] = tag
    conn.close()
    return obs_arr, mask_arr, pol_arr, val_arr, tag_arr


# ── Tag utilities ───────────────────────────────────────────────────────────

def tag_display(tag):
    if tag is None or tag == "" or tag == "None":
        return "normal"
    return str(tag)


def build_filtered_indices(tags, total, tag_filter):
    """Return array of indices matching *tag_filter*."""
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


class TagFilter:
    """Mutable filter state + cursor position for buffer browsing."""

    def __init__(self, tags_arr, initial_filter="all"):
        self._tags = tags_arr
        self._name = initial_filter
        self._indices = build_filtered_indices(tags_arr,
                                               len(tags_arr) if tags_arr is not None else 0,
                                               initial_filter)
        self._pos = 0

    @property
    def name(self):
        return self._name

    @property
    def indices(self):
        return self._indices

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        n = len(self._indices)
        self._pos = value % n if n > 0 else 0

    def set_filter(self, name):
        self._name = name
        total = len(self._tags) if self._tags is not None else 0
        self._indices = build_filtered_indices(self._tags, total, name)
        self._pos = 0

    @property
    def current_global_idx(self):
        if len(self._indices) == 0:
            return -1
        return int(self._indices[self._pos % len(self._indices)])

    def next(self, delta=1):
        n = len(self._indices)
        if n > 0:
            self._pos = (self._pos + delta) % n

    def prev(self, delta=1):
        self.next(-delta)

    def goto_first(self):
        self._pos = 0

    def goto_last(self):
        n = len(self._indices)
        if n > 0:
            self._pos = n - 1


# ── Shared font caching ─────────────────────────────────────────────────────

_FONT_CACHE = {}

def get_font(size):
    """Return a cached pygame Font."""
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.Font(None, size)
    return _FONT_CACHE[size]


# ── Shared heatmap surface cache ────────────────────────────────────────────

_HEATMAP_CACHE = {}

def heatmap_surface(sq_size, intensity):
    """Return a cached transparent Surface for heatmap overlays."""
    key = (sq_size, int(intensity * 100))
    if key not in _HEATMAP_CACHE:
        r, g, b = int(255 * intensity), int(128 * (1 - intensity)), int(128 * (1 - intensity))
        s = pygame.Surface((sq_size, sq_size), pygame.SRCALPHA)
        s.fill((r, g, b, 120))
        _HEATMAP_CACHE[key] = s
    return _HEATMAP_CACHE[key]
