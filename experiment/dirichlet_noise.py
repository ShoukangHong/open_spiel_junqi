"""Visualise Dirichlet noise samples with matplotlib.

Edit the parameters in __main__ and run.
"""
import matplotlib.pyplot as plt
import numpy as np


def show_noise(alpha=1.0, num_actions=8, epsilon=0.25, samples=2000):
    """Generate Dirichlet noise and plot distributions.

    Args:
        alpha: Dirichlet concentration parameter (same for all actions).
        num_actions: number of legal actions (N).
        epsilon: noise mixing weight (blended prior = (1-ε)×prior + ε×noise).
        samples: number of Dirichlet draws.
    """
    rng = np.random.default_rng()
    noise = rng.dirichlet([alpha] * num_actions, samples)

    mean = noise.mean(axis=0)
    uniform_prior = 1.0 / num_actions
    blended = (1 - epsilon) * uniform_prior + epsilon * noise
    blended_top = blended.max(axis=1)

    # ── Figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Dirichlet  α={alpha}  N={num_actions}  ε={epsilon}  ×{samples}")

    # 1. Per-action noise distribution (box plot)
    ax = axes[0, 0]
    ax.boxplot(noise, positions=range(num_actions), widths=0.6,
               flierprops={"marker": ".", "markersize": 2})
    ax.axhline(y=uniform_prior, color="gray", linestyle="--", label=f"uniform={uniform_prior:.3f}")
    ax.set_xlabel("action index")
    ax.set_ylabel("noise value")
    ax.set_title("Per-action noise distribution")
    ax.legend()

    # 2. Blended prior (histogram)
    ax = axes[0, 1]
    ax.hist(blended.ravel(), bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=uniform_prior, color="gray", linestyle="--", label=f"uniform={uniform_prior:.3f}")
    ax.axvline(x=blended_top.mean(), color="red", linestyle="-",
               label=f"top action mean={blended_top.mean():.3f}")
    ax.set_xlabel("blended prior value")
    ax.set_ylabel("count")
    ax.set_title(f"Blended prior = (1-{epsilon})×uniform + {epsilon}×noise")
    ax.legend()

    # 3. Example draws (first 3 samples as bar charts)
    ax = axes[1, 0]
    x = np.arange(num_actions)
    colors = ["steelblue", "darkorange", "seagreen"]
    for i, c in zip(range(3), colors):
        ax.bar(x + i * 0.25, noise[i], width=0.25, color=c, alpha=0.7,
               label=f"sample {i+1}")
    ax.axhline(y=uniform_prior, color="gray", linestyle="--")
    ax.set_xlabel("action index")
    ax.set_ylabel("noise value")
    ax.set_title("Three example Dirichlet draws")
    ax.legend()

    # 4. Entropy histogram
    ax = axes[1, 1]
    eps = 1e-12
    ent = -(noise * np.log(noise + eps)).sum(axis=1)
    uniform_ent = np.log(num_actions)
    ax.hist(ent, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=uniform_ent, color="gray", linestyle="--", label=f"uniform={uniform_ent:.3f}")
    ax.axvline(x=ent.mean(), color="red", linestyle="-", label=f"mean={ent.mean():.3f}")
    ax.set_xlabel("entropy (nats)")
    ax.set_ylabel("count")
    ax.set_title(f"Entropy distribution  (eff={ent.mean()/uniform_ent*100:.0f}%)")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # ── Console summary ────────────────────────────────────────────────
    flat = noise.ravel()
    print(f"Dirichlet(α={alpha}) × N={num_actions}, ε={epsilon}")
    print(f"  noise:    mean={mean.mean():.3f}  std={mean.std():.3f}  "
          f"range=[{flat.min():.4f}, {flat.max():.3f}]")
    print(f"  blended:  top action mean={blended_top.mean():.3f}  "
          f"boost={(blended_top.mean()/uniform_prior - 1)*100:.0f}%")
    print(f"  entropy:  {ent.mean():.3f} / {uniform_ent:.3f}  "
          f"efficiency={ent.mean()/uniform_ent*100:.0f}%")
    print(f"  α≈10/N = {10/num_actions:.2f}")


if __name__ == "__main__":
    # ── Edit parameters here ───────────────────────────────────────────
    ALPHA = 0.8       # Dirichlet concentration
    ACTIONS = 12       # number of legal actions (Othello avg)
    EPSILON = 0.25    # noise mixing weight
    SAMPLES = 2000    # number of draws

    show_noise(alpha=ALPHA, num_actions=ACTIONS, epsilon=EPSILON,
               samples=SAMPLES)
