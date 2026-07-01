"""Compare surprise detection on bugchild vs bugchild_2 — focus on outcome."""
import pyspiel, numpy as np, os, sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

CKPT_DIR = r"C:\Users\shouk\xiangqi_train\cloud_new"
S1 = os.path.join(CKPT_DIR, "saved_positions", "bugchild.txt")
S2 = os.path.join(CKPT_DIR, "saved_positions", "bugchild_2.txt")

from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.mcts import BatchMCTS, compute_solved_policy
from train.batch_mcts.evaluator import PyTorchEvaluator
from train.model.xiangqi_resnet import XiangqiResNet
from train.model.model import Model
from train.core.surprise import detect_surprise, _stable_qdr, _kl, _wdl_from_qdr

game = pyspiel.load_game("xiangqi")
resnet = XiangqiResNet(input_channels=17, board_rows=10, board_cols=9,
                        output_size=8100, nn_width=160, nn_depth=10)
model = Model(resnet, learning_rate=1e-3, weight_decay=1e-4, device="cpu")
model.load_checkpoint(70)
ev = PyTorchEvaluator(game, model)

mcts_cfg = MCTSConfig(
    max_simulations=600, batch_size=10,
    uct_c=3.0, policy_epsilon=0.25, policy_alpha=0.33,
    draw_penalty=0.1, repeat_penalty=0.2, fpu_lambda=0.2,
    probe_depth=1, probe_surprise=0.3, verbose=False)

from types import SimpleNamespace
s_cfg = SimpleNamespace(
    surprise_pol_kl=0.05, surprise_val_kl=0.05,  # very low to see all
    surprise_child_min_n=1)

for label, path in [("bugchild", S1), ("bugchild_2", S2)]:
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    with open(path) as f:
        state = game.deserialize_state(f.read().strip())
    player = state.current_player()
    t = np.array(state.observation_tensor()).reshape(17, 10, 9)
    print(f"Player: {player}  No-capture: {t[16,0,0]*40:.1f}/40  "
          f"Legal actions: {len(state.legal_actions())}")

    mcts = BatchMCTS(game, mcts_cfg, ev, random_state=np.random.RandomState(42))
    root = mcts.mcts_search(state)
    print(f"Root: N={root.explore_count}  Q={root.q_value:+.4f}  dr={root.draw_rate:.4f}")

    tag, child_tags, combined = detect_surprise(state, root, s_cfg, game.max_utility())
    print(f"Root tag: '{tag}'  combined={combined:.4f}")
    print(f"Child tags: {len(child_tags)}")

    if child_tags:
        print(f"\n{'action':>8} {'tag':<28} {'cq':>8} {'cdr':>8} {'c.N':>5} "
              f"{'nn_q':>8} {'nn_dr':>8} {'outcome?':>10} {'c.state':>10}")
        print("-" * 110)
        for a, (ctag, ckl) in sorted(child_tags.items(), key=lambda x: x[1][1], reverse=True):
            c = next(c for c in root.children if c.action == a)
            cq, cdr = _stable_qdr(c)
            has_o = str(c.outcome.tolist()) if c.outcome is not None else "None"
            has_st = "TERM" if c.state and c.state.is_terminal() else "OK"
            nn_q = c.nn_q if c.nn_q is not None else float('nan')
            nn_dr = c.nn_draw if c.nn_draw is not None else float('nan')

            # Manual KL breakdow
            m_wdl = _wdl_from_qdr(cq, cdr)
            n_wdl = _wdl_from_qdr(nn_q, nn_dr) if not np.isnan(nn_q) else None
            vkl = _kl(m_wdl, n_wdl) if n_wdl is not None else float('nan')

            print(f"  {a:>6d}  {ctag:<28s} {cq:+8.4f} {cdr:+8.4f} {c.explore_count:5d}  "
                  f"{nn_q:+8.4f} {nn_dr:+8.4f}  {has_o:>10s} {has_st:>10s}  "
                  f"m_wdl={m_wdl}  n_wdl={n_wdl}")

    # Also show children with extreme Q values (proven win/loss candidate)
    print(f"\n  Children with proven outcome:")
    proven = [c for c in root.children if c.outcome is not None]
    for c in proven[:5]:
        cq, cdr = _stable_qdr(c)
        nn_q = c.nn_q
        nn_dr = c.nn_draw or 0.0
        m_wdl = _wdl_from_qdr(cq, cdr)
        n_wdl = _wdl_from_qdr(nn_q, nn_dr) if nn_q is not None else None
        print(f"    a={c.action} outcome={c.outcome.tolist()}  state.player={c.state.current_player() if c.state else 'NONE'}  "
              f"cq={cq:+.4f}  nn_q={nn_q:+.4f}  m_wdl={m_wdl}  n_wdl={n_wdl}")
