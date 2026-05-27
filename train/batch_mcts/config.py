"""MCTS configuration — all hyperparameters in one place."""

from dataclasses import dataclass


@dataclass
class MCTSConfig:
    """Hyperparameters for Batch MCTS."""

    # UCT
    uct_c: float = 1.41         # PUCT exploration constant

    # Simulation budget
    max_simulations: int = 200  # Total MCTS simulations per move
    batch_size: int = 32        # Number of virtual threads per batch

    # Dirichlet noise (AlphaZero)
    policy_epsilon: float = 0.25  # Mixing weight
    policy_alpha: float = 1.0     # Concentration parameter

    # Virtual loss
    virtual_loss: float = 1.0   # Added during selection, removed at backprop

    # Action selection
    temperature: float = 1.0     # Softmax temperature for move selection
    temperature_drop: int = 10   # Move after which to switch to argmax

    # Solver
    solve: bool = True           # Whether to back up proven terminal values

    # Value head format
    value_classes: int = 1       # 1 = scalar [-1,1]; 3 = WDL softmax

    # Debug
    verbose: bool = False
