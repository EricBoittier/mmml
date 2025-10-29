# ================================================
# Neural-guided MCTS with JAX + Flax (TicTacToe)
# ================================================
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# ---------- JAX / Flax ----------
import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial

# =========================================================
# Minimal TicTacToe env (same API as before)
# =========================================================
class TicTacToe:
    """
    Board: 3x3 np.int8
      1 -> 'X', -1 -> 'O', 0 -> empty
    Player to move: 1 or -1
    """
    def __init__(self, board: Optional[np.ndarray] = None, to_move: int = 1):
        self.board = np.zeros((3,3), dtype=np.int8) if board is None else board.copy()
        self.to_move = int(to_move)

    def clone(self): return TicTacToe(self.board, self.to_move)

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.board.ravel() == 0)

    def step(self, action: int) -> "TicTacToe":
        r, c = divmod(int(action), 3)
        if self.board[r, c] != 0: raise ValueError("Illegal action")
        nxt = self.clone()
        nxt.board[r, c] = self.to_move
        nxt.to_move = -self.to_move
        return nxt

    def is_terminal(self) -> bool:
        return self.winner() is not None or self.legal_actions().size == 0

    def winner(self) -> Optional[int]:
        b = self.board
        lines = list(b) + list(b.T) + [np.diag(b), np.diag(np.fliplr(b))]
        for line in lines:
            s = int(np.sum(line))
            if s == 3:  return 1
            if s == -3: return -1
        if np.all(b != 0): return 0
        return None

    def result_from(self, player: int) -> float:
        w = self.winner()
        if w is None: raise ValueError("Not terminal")
        if w == 0: return 0.0
        return 1.0 if w == player else -1.0

    def __repr__(self) -> str:
        s = {1:"X",-1:"O",0:"."}
        rows = [" ".join(s[int(x)] for x in row) for row in self.board]
        return "\n".join(rows) + f"\nTo move: {'X' if self.to_move==1 else 'O'}"

# =========================================================
# AlphaZero-style PUCT MCTS (unchanged)
# =========================================================
@dataclass
class Node:
    key: Tuple
    to_move: int
    parent: Optional["Node"] = None
    parent_action: Optional[int] = None

    children: Dict[int, "Node"] = field(default_factory=dict)  # action -> child node
    P: Dict[int, float] = field(default_factory=dict)          # action -> prior prob

    N: int = 0
    W: float = 0.0
    Q: float = 0.0

    def expanded(self) -> bool:
        return len(self.P) > 0

def state_to_key(env: TicTacToe) -> Tuple:
    return (tuple(env.board.ravel().tolist()), env.to_move)

class PUCT_MCTS:
    """
    Selection: a* = argmax[ Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a)) ]
    Expansion: add new child with policy prior
    Evaluation: value network v(s) (from current player POV)
    Backup: update stats with conversion to root POV
    """
    def __init__(self, policy_value_fn, c_puct: float = 1.5, dirichlet_alpha: float = 0.3,
                 root_noise_frac: float = 0.25, rng: Optional[np.random.Generator] = None):
        self.policy_value_fn = policy_value_fn
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.root_noise_frac = float(root_noise_frac)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_root = None

    def search(self, root_env: TicTacToe, n_simulations: int = 400, temperature: float = 1.0) -> int:
        root = Node(key=state_to_key(root_env), to_move=root_env.to_move)
        self._expand(root_env, root)
        self._add_root_dirichlet_noise(root)

        for _ in range(n_simulations):
            self._simulate(root_env, root, root_player=root_env.to_move)
    
        self.last_root = root
        return self._select_action_from_visits(root, temperature)


    def _simulate(self, root_env: TicTacToe, root: Node, root_player: int):
        node = root
        env = root_env.clone()
        path = []

        while True:
            if env.is_terminal():
                value = env.result_from(root_player)
                self._backup(path, leaf_value=value)
                return

            a = self._puct_select(node)
            path.append((node, a))

            if a not in node.children:
                env = env.step(a)
                child = Node(key=state_to_key(env), to_move=env.to_move, parent=node, parent_action=a)
                priors, leaf_val_from_to_move = self.policy_value_fn(env)
                child.P = priors
                node.children[a] = child
                self._backup(path, leaf_value=leaf_val_from_to_move, leaf_env_to_move=env.to_move, root_player=root_player)
                return

            env = env.step(a)
            node = node.children[a]

    def _puct_select(self, node: Node) -> int:
        sum_N = max(1, sum(child.N for child in node.children.values()))
        best_a, best_score = None, -1e18
        for a, p in node.P.items():
            child = node.children.get(a)
            Nsa = 0 if child is None else child.N
            Qsa = 0.0 if child is None else child.Q
            u = self.c_puct * p * np.sqrt(sum_N) / (1.0 + Nsa)
            score = Qsa + u
            if score > best_score:
                best_score = score
                best_a = a
        return int(best_a)

    def _expand(self, env: TicTacToe, node: Node):
        priors, _ = self.policy_value_fn(env)
        node.P = priors

    def _add_root_dirichlet_noise(self, root: Node):
        if not root.P: return
        actions = list(root.P.keys())
        alpha = self.dirichlet_alpha
        noise = self.rng.dirichlet([alpha] * len(actions))
        for a, eps in zip(actions, noise):
            root.P[a] = (1 - self.root_noise_frac) * root.P[a] + self.root_noise_frac * float(eps)

    def _backup(self, path, leaf_value: float, leaf_env_to_move: Optional[int] = None, root_player: Optional[int] = None):
        if leaf_env_to_move is None or root_player is None:
            v_root = leaf_value
            for node, _ in reversed(path):
                node.N += 1; node.W += v_root; node.Q = node.W / node.N
                v_root = -v_root
            return

        same = 1.0 if leaf_env_to_move == root_player else -1.0
        v_root = leaf_value * same
        for node, _ in reversed(path):
            node.N += 1; node.W += v_root; node.Q = node.W / node.N
            v_root = -v_root

    def _select_action_from_visits(self, root: Node, temperature: float) -> int:
        acts = np.array(list(root.P.keys()), dtype=int)
        visits = np.array([root.children[a].N if a in root.children else 0 for a in acts], dtype=float)
        if temperature <= 1e-8:
            return int(acts[np.argmax(visits)])
        with np.errstate(divide='ignore', invalid='ignore'):
            pi = np.power(visits, 1.0 / temperature)
        if np.all(pi == 0): pi = np.ones_like(pi)
        pi = pi / np.sum(pi)
        return int(np.random.default_rng().choice(acts, p=pi))

# =========================================================
# Flax policy/value network
# =========================================================
def make_features(board_np: np.ndarray, to_move: int) -> jnp.ndarray:
    """
    Returns (3,3,3) float32:
      ch0 = 1 where X stones
      ch1 = 1 where O stones
      ch2 = to_move (all +1 if X to move, all -1 if O to move)
    """
    b = board_np.astype(np.int8)
    x = (b == 1).astype(np.float32)
    o = (b == -1).astype(np.float32)
    tm = np.full_like(x, float(1.0 if to_move == 1 else -1.0), dtype=np.float32)
    feat = np.stack([x, o, tm], axis=-1)  # (3,3,3)
    return jnp.asarray(feat)

class TTTNet(nn.Module):
    """Tiny conv net with separate policy and value heads."""
    @nn.compact
    def __call__(self, x):  # x: (B,3,3,3)
        # Trunk
        y = nn.Conv(features=32, kernel_size=(3,3), padding='SAME')(x)
        y = nn.relu(y)
        y = nn.Conv(features=64, kernel_size=(3,3), padding='SAME')(y)
        y = nn.relu(y)
        y = y.reshape((y.shape[0], -1))
        y = nn.Dense(64)(y); y = nn.relu(y)

        # Policy head -> logits for 9 actions
        p = nn.Dense(32)(y); p = nn.relu(p)
        policy_logits = nn.Dense(9)(p)  # (B,9)

        # Value head -> scalar in [-1,1]
        v = nn.Dense(32)(y); v = nn.relu(v)
        value = nn.Dense(1)(v)
        value = nn.tanh(value)          # (B,1)
        return policy_logits, value[:, 0]

def mask_and_softmax(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    """Softmax over legal actions only. logits, mask shape (9,), mask {0,1}."""
    # Set illegal to a large negative.
    masked = jnp.where(legal_mask > 0.5, logits, -1e9)
    # Avoid NaNs if all illegal (shouldn't happen)
    masked = jnp.nan_to_num(masked, nan=-1e9)
    exps = jnp.exp(masked - jnp.max(masked))
    denom = jnp.sum(exps)
    probs = jnp.where(denom > 0, exps / denom, jnp.ones_like(exps) / exps.shape[0])
    return probs

@partial(jax.jit, static_argnums=0)
def model_apply_jit(model: TTTNet, params, feats: jnp.ndarray, legal_mask: jnp.ndarray):
    """Batched apply; feats (B,3,3,3), legal_mask (B,9)."""
    logits, value = model.apply(params, feats)      # logits (B,9), value (B,)
    probs = jax.vmap(mask_and_softmax)(logits, legal_mask)
    return probs, value

# Wrapper to create a policy_value_fn compatible with PUCT_MCTS
def make_flax_policy_value_fn(model: TTTNet, params):
    rng = np.random.default_rng()

    def policy_value_fn(env: TicTacToe):
        legal = env.legal_actions()
        # Build feature tensor (B=1)
        feats = make_features(env.board, env.to_move)[None, ...]   # (1,3,3,3)
        # Build legal mask (1,9)
        mask = np.zeros((1, 9), dtype=np.float32)
        mask[0, legal] = 1.0
        probs_b, value_b = model_apply_jit(model, params, feats, jnp.asarray(mask))
        probs = np.array(probs_b[0])   # (9,)
        value = float(np.array(value_b[0]))  # from CURRENT to-move POV (as desired)

        # Convert to dict over legal actions only
        priors = {int(a): float(probs[a]) for a in legal}
        # Normalize just in case of tiny numerical drift
        s = sum(priors.values())
        if s <= 0:
            # Fallback uniform
            u = 1.0 / max(1, len(legal))
            priors = {int(a): u for a in legal}
        else:
            for a in list(priors.keys()):
                priors[a] /= s
        return priors, value

    return policy_value_fn




# ============================
# Self-play & Replay Buffer
# ============================
"""
Monte Carlo Tree Search (MCTS) implementation for Tic-Tac-Toe with neural network guidance.

This module implements:
- ReplayBuffer: Stores training samples from self-play games
- Self-play: Generates training data using MCTS with neural network guidance
- Training: JAX-based training loop for policy and value networks
- Batched evaluation: Efficient evaluation of multiple game positions

Key concepts:
- Self-play: The agent plays against itself to generate training data
- MCTS: Monte Carlo Tree Search for decision making during self-play
- Policy network: Predicts move probabilities
- Value network: Predicts game outcome probabilities
- Replay buffer: Stores (state, policy, value) tuples for training

Example usage:
    # Initialize components
    model = TTTNet()
    mcts = PUCT_MCTS(policy_value_fn=pv_fn, c_puct=1.5)
    buffer = ReplayBuffer(capacity=10000)
    
    # Generate training data through self-play
    samples = self_play_game(mcts, model, params, temperature=1.0)
    buffer.add_many(samples)
    
    # Train the model
    params = train_loop(num_iters=50, games_per_iter=16)
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
from typing import List, Tuple, Dict

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, rng: np.random.Generator = None):
        self.capacity = capacity
        self.rng = rng or np.random.default_rng()
        self._data = []  # list of (feat: (3,3,3), pi: (9,), z: scalar)

    def add_many(self, samples: List[Tuple[np.ndarray, np.ndarray, float]]):
        if not samples: return
        self._data.extend(samples)
        if len(self._data) > self.capacity:
            # Drop oldest
            self._data = self._data[-self.capacity:]

    def sample(self, batch_size: int):
        idx = self.rng.choice(len(self._data), size=batch_size, replace=False)
        feats = np.stack([self._data[i][0] for i in idx], axis=0).astype(np.float32)   # (B,3,3,3)
        pis   = np.stack([self._data[i][1] for i in idx], axis=0).astype(np.float32)   # (B,9)
        zs    = np.array([self._data[i][2] for i in idx], dtype=np.float32)            # (B,)
        return feats, pis, zs

    def __len__(self):
        return len(self._data)

def visits_to_pi(root_node) -> np.ndarray:
    """
    Convert MCTS visit counts to a probability distribution over actions.
    
    This function takes the visit counts from MCTS tree search and converts them
    into a probability distribution that can be used as training targets for the
    policy network. Actions with more visits get higher probabilities.
    
    Args:
        root_node: MCTS root node with children containing visit counts
        
    Returns:
        Probability distribution over 9 actions (3x3 board positions)
        
    Example:
        >>> # After MCTS search, root node has children with visit counts
        >>> root.children = {0: Node(N=10), 1: Node(N=5), 2: Node(N=15)}
        >>> pi = visits_to_pi(root)
        >>> print(pi)  # [0.33, 0.17, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        >>> print(f"Total probability: {pi.sum()}")  # Should be 1.0
    """
    pi = np.zeros(9, dtype=np.float32)
    total = 0
    for a, child in root_node.children.items():
        n = child.N
        pi[a] = n
        total += n
    if total > 0:
        pi /= total
    else:
        # fallback uniform distribution if no visits
        pi[:] = 1.0 / 9.0
    return pi

def outcome_to_value_for_player(winner: int, player: int) -> float:
    """
    Convert game outcome to value from a specific player's perspective.
    
    This function maps the game result to a value that represents how good
    the outcome is for the given player. Used for creating training targets
    for the value network.
    
    Args:
        winner: Game winner (-1 for player -1, 0 for draw, 1 for player 1)
        player: Player whose perspective to take (-1 or 1)
        
    Returns:
        Value from player's perspective:
        - 1.0: Player won
        - 0.0: Draw
        - -1.0: Player lost
        
    Example:
        >>> # Player 1 wins
        >>> value = outcome_to_value_for_player(winner=1, player=1)
        >>> print(value)  # 1.0
        
        >>> # Player -1 wins, but we're evaluating from player 1's perspective
        >>> value = outcome_to_value_for_player(winner=-1, player=1)
        >>> print(value)  # -1.0
        
        >>> # Draw game
        >>> value = outcome_to_value_for_player(winner=0, player=1)
        >>> print(value)  # 0.0
    """
    if winner == 0: return 0.0
    return 1.0 if winner == player else -1.0

def self_play_game(mcts, model, params, temperature: float = 1.0, temp_moves: int = 6):
    """
    Play one complete game using MCTS with neural network guidance.
    
    This function implements self-play where the agent plays against itself using
    MCTS for move selection. The neural network provides policy and value estimates
    to guide the search. Temperature controls exploration vs exploitation.
    
    Args:
        mcts: MCTS instance configured with policy-value function
        model: Neural network model (TTTNet)
        params: Current model parameters
        temperature: Controls move selection randomness (default: 1.0)
        temp_moves: Number of moves to use high temperature (default: 6)
        
    Returns:
        List of (features, policy, value) tuples for training:
        - features: (3,3,3) game state representation
        - policy: (9,) move probability distribution from MCTS visits
        - value: Game outcome from each position's player perspective
        
    Example:
        >>> # Initialize MCTS and model
        >>> mcts = PUCT_MCTS(policy_value_fn=pv_fn, c_puct=1.5)
        >>> model = TTTNet()
        >>> params = model.init(key, dummy_input)
        >>> 
        >>> # Play a game with high exploration early, then greedy
        >>> samples = self_play_game(mcts, model, params, temperature=1.0, temp_moves=6)
        >>> print(f"Game generated {len(samples)} training samples")
        >>> 
        >>> # Each sample contains state, policy, and value
        >>> for i, (feat, pi, z) in enumerate(samples):
        ...     print(f"Move {i}: value={z:.2f}, best_move={pi.argmax()}")
    """
    from copy import deepcopy
    env = TicTacToe()
    samples = []

    move_idx = 0
    states_pov: List[int] = []  # player to move for each recorded state (for value target)
    while not env.is_terminal():
        # Make features & run MCTS
        feats = make_features(env.board, env.to_move)  # (3,3,3)
        tau = temperature if move_idx < temp_moves else 1e-9

        # Run MCTS with current params (wrap a flax policy/value fn)
        pv_fn = make_flax_policy_value_fn(model, params)
        action = mcts.search(env, n_simulations=200, temperature=tau)  # you can tune sims

        # Grab pi from the root (visit counts distribution)
        # The PUCT_MCTS doesn't return the root; quick hack: rerun expand to reconstruct priors,
        # then get visits from the internal root held during the last search. Easiest approach:
        # modify search() to return both action and the root node. For brevity, we just recompute here:
        # (Instead, we store it inside mcts for this call.)
        # ---- Small patch: add a field 'last_root' on mcts in its search() implementation ----
        pi = visits_to_pi(mcts.last_root)

        samples.append((np.array(feats, dtype=np.float32), pi.copy(), 0.0))  # z is filled later
        states_pov.append(env.to_move)

        # Play move
        env = env.step(int(action))
        move_idx += 1

    # Game finished; set z for all samples
    w = env.winner()  # in {-1,0,1}
    finalized = []
    for (feat, pi, _), pov in zip(samples, states_pov):
        z = outcome_to_value_for_player(w, pov)
        finalized.append((feat, pi, z))
    return finalized
# ============================
# Training step (JAX/Optax)
# ============================
from flax.core import FrozenDict

def masked_log_softmax(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Compute log softmax over legal moves only.
    
    This function applies log softmax to the logits but masks out illegal moves
    by setting their logits to a very negative value (-1e9), effectively
    giving them zero probability.
    
    Args:
        logits: Raw network outputs (B, 9) for 9 possible moves
        legal_mask: Binary mask (B, 9) where 1=legal, 0=illegal
        
    Returns:
        Log probabilities (B, 9) with illegal moves masked out
        
    Example:
        >>> logits = jnp.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        >>> legal_mask = jnp.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        >>> log_probs = masked_log_softmax(logits, legal_mask)
        >>> print(log_probs)  # Only first 3 moves have non-zero log probs
    """
    masked = jnp.where(legal_mask > 0.5, logits, -1e9)
    return jax.nn.log_softmax(masked, axis=-1)

def make_legal_mask_from_feats(feats: jnp.ndarray) -> jnp.ndarray:
    """
    Create legal move mask from game state features.
    
    For TicTacToe, a move is legal if the corresponding cell is empty
    (both X and O channels are 0).
    
    Args:
        feats: Game state features (B, 3, 3, 3) with channels [X, O, to_move]
        
    Returns:
        Legal move mask (B, 9) where 1=legal, 0=illegal
        
    Example:
        >>> # Empty board - all moves legal
        >>> feats = jnp.zeros((1, 3, 3, 3))
        >>> mask = make_legal_mask_from_feats(feats)
        >>> print(mask)  # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        >>> 
        >>> # Center occupied - center move illegal
        >>> feats = feats.at[0, 1, 1, 0].set(1.0)  # X in center
        >>> mask = make_legal_mask_from_feats(feats)
        >>> print(mask[0, 4])  # 0.0 (center position is illegal)
    """
    x = feats[..., 0]
    o = feats[..., 1]
    empty = (x == 0) & (o == 0)
    return empty.reshape((empty.shape[0], 9)).astype(jnp.float32)


from functools import partial

@partial(jax.jit, static_argnames=['model'])
def forward_for_train(params: FrozenDict, model: TTTNet, feats: jnp.ndarray):
    """
    Forward pass through the neural network for training.
    
    Returns raw logits and value predictions without applying softmax or masking.
    The training loss function handles masking and probability conversion.
    
    Args:
        params: Model parameters
        model: Neural network model (TTTNet)
        feats: Game state features (B, 3, 3, 3)
        
    Returns:
        Tuple of (policy_logits, value):
        - policy_logits: (B, 9) raw logits for 9 possible moves
        - value: (B,) predicted game outcome probabilities
        
    Example:
        >>> logits, value = forward_for_train(params, model, feats)
        >>> print(f"Logits shape: {logits.shape}")  # (batch_size, 9)
        >>> print(f"Value shape: {value.shape}")    # (batch_size,)
    """
    logits, value = model.apply(params, feats)
    return logits, value  # (B,9), (B,)

def policy_value_loss(params: FrozenDict, model: TTTNet,
                      feats: jnp.ndarray, target_pi: jnp.ndarray, target_z: jnp.ndarray,
                      weight_policy: float = 1.0, weight_value: float = 1.0):
    """
    Compute combined policy and value loss for training.
    
    This function implements the AlphaZero-style loss combining:
    - Policy loss: Cross-entropy between predicted and target move probabilities
    - Value loss: Mean squared error between predicted and actual game outcomes
    
    Args:
        params: Model parameters
        model: Neural network model (TTTNet)
        feats: Game state features (B, 3, 3, 3)
        target_pi: Target move probabilities from MCTS (B, 9)
        target_z: Target game outcomes (B,)
        weight_policy: Weight for policy loss component (default: 1.0)
        weight_value: Weight for value loss component (default: 1.0)
        
    Returns:
        Tuple of (total_loss, metrics_dict):
        - total_loss: Combined weighted loss
        - metrics_dict: Dictionary with individual loss components and MAE
        
    Example:
        >>> # Training step
        >>> loss, metrics = policy_value_loss(params, model, feats, target_pi, target_z)
        >>> print(f"Total loss: {loss:.4f}")
        >>> print(f"Policy loss: {metrics['policy_loss']:.4f}")
        >>> print(f"Value loss: {metrics['value_loss']:.4f}")
        >>> print(f"Value MAE: {metrics['value_mae']:.4f}")
    """
    logits, value = forward_for_train(params, model, feats)  # (B,9), (B,)
    legal_mask = jnp.asarray(make_legal_mask_from_feats(jnp.array(feats)))  # (B,9) on host -> device

    logp = masked_log_softmax(logits, legal_mask)  # (B,9)
    # Cross-entropy loss: -sum(target_pi * log_predicted_pi)
    pol_loss = -jnp.mean(jnp.sum(target_pi * logp, axis=-1))
    val_loss = jnp.mean((value - target_z) ** 2)
    loss = weight_policy * pol_loss + weight_value * val_loss
    metrics = {
        "loss": loss,
        "policy_loss": pol_loss,
        "value_loss": val_loss,
        "value_mae": jnp.mean(jnp.abs(value - target_z)),
    }
    return loss, metrics

def make_optimizer(lr: float = 1e-3, weight_decay: float = 1e-4):
    """
    Create AdamW optimizer for training.
    
    Args:
        lr: Learning rate (default: 1e-3)
        weight_decay: Weight decay for L2 regularization (default: 1e-4)
        
    Returns:
        Optax optimizer instance
        
    Example:
        >>> optimizer = make_optimizer(lr=0.001, weight_decay=1e-4)
        >>> opt_state = optimizer.init(params)
    """
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    return tx

from functools import partial

@partial(jax.jit, static_argnames=['model', 'tx'])
def train_step(params: FrozenDict, opt_state, model: TTTNet,
               feats: jnp.ndarray, target_pi: jnp.ndarray, target_z: jnp.ndarray, tx):
    """
    Perform one training step with gradient descent.
    
    This function computes gradients, updates parameters, and returns training metrics.
    It's JIT-compiled for efficiency.
    
    Args:
        params: Current model parameters
        opt_state: Optimizer state
        model: Neural network model (TTTNet)
        feats: Game state features (B, 3, 3, 3)
        target_pi: Target move probabilities (B, 9)
        target_z: Target game outcomes (B,)
        tx: Optax optimizer
        
    Returns:
        Tuple of (updated_params, updated_opt_state, metrics_dict):
        - updated_params: Parameters after gradient update
        - updated_opt_state: Optimizer state after update
        - metrics_dict: Training metrics (loss, policy_loss, value_loss, value_mae)
        
    Example:
        >>> # Single training step
        >>> params, opt_state, metrics = train_step(
        ...     params, opt_state, model, feats, target_pi, target_z, optimizer
        ... )
        >>> print(f"Loss: {metrics['loss']:.4f}")
    """
    (loss, metrics), grads = jax.value_and_grad(policy_value_loss, has_aux=True)(
        params, model, feats, target_pi, target_z
    )
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, metrics



# ============================
# Simple training loop
# ============================
def train_loop(num_iters=50, games_per_iter=16, batch_size=64, train_steps=100,
               sims_per_move=200, lr=1e-3, wd=1e-4, seed=0):
    """
    Complete AlphaZero-style training loop for Tic-Tac-Toe.
    
    This function implements the full training pipeline:
    1. Initialize model and MCTS
    2. For each iteration:
       - Generate training data through self-play
       - Train the neural network on collected data
    3. Return trained parameters
    
    The training follows the AlphaZero algorithm:
    - Self-play generates (state, policy, value) samples
    - Neural network learns to predict MCTS policy and game outcomes
    - Improved network guides better MCTS searches
    
    Args:
        num_iters: Number of training iterations (default: 50)
        games_per_iter: Games to play per iteration (default: 16)
        batch_size: Training batch size (default: 64)
        train_steps: Training steps per iteration (default: 100)
        sims_per_move: MCTS simulations per move (default: 200)
        lr: Learning rate (default: 1e-3)
        wd: Weight decay (default: 1e-4)
        seed: Random seed for reproducibility (default: 0)
        
    Returns:
        Trained model parameters
        
    Example:
        >>> # Train a Tic-Tac-Toe agent
        >>> params = train_loop(
        ...     num_iters=10,
        ...     games_per_iter=8,
        ...     batch_size=32,
        ...     train_steps=50,
        ...     lr=0.001
        ... )
        >>> print("Training completed!")
        >>> 
        >>> # Monitor training progress
        >>> # The function prints loss metrics every 25 steps
        >>> # [iter 1 step 25] loss=2.1234 pol=1.5678 val=0.5556 mae=0.234
    """
    rng = np.random.default_rng(seed)

    # Init model + params
    model = TTTNet()
    key = jax.random.PRNGKey(seed)
    dummy_feats = jnp.zeros((1,3,3,3), dtype=jnp.float32)
    params = model.init(key, dummy_feats)

    # MCTS with neural guidance
    pv_fn = make_flax_policy_value_fn(model, params)
    mcts = PUCT_MCTS(
        policy_value_fn=pv_fn,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        root_noise_frac=0.25,
        rng=rng
    )

    # PATCH: store root in search() (add to PUCT_MCTS.search before returning):
    #   self.last_root = root
    # And use sims_per_move arg inside search
    # (to keep snippet focused, we assume you've added it)

    rb = ReplayBuffer(capacity=50_000, rng=rng)
    tx = make_optimizer(lr=lr, weight_decay=wd)
    opt_state = tx.init(params)

    for it in range(1, num_iters + 1):
        # ---- Self-play ----
        all_samples = []
        for _ in range(games_per_iter):
            pv_fn = make_flax_policy_value_fn(model, params)  # refresh with latest params
            mcts.policy_value_fn = pv_fn
            samples = self_play_game(mcts, model, params, temperature=1.0, temp_moves=6)
            all_samples.extend(samples)
        rb.add_many(all_samples)

        # ---- Training ----
        if len(rb) < batch_size:
            print(f"[iter {it}] warming up buffer ({len(rb)} samples)")
            continue

        for step in range(train_steps):
            feats, pis, zs = rb.sample(batch_size)
            params, opt_state, metrics = train_step(
                params, opt_state, model,
                jnp.asarray(feats), jnp.asarray(pis), jnp.asarray(zs),
                tx
            )
            if (step + 1) % 25 == 0:
                print(f"[iter {it} step {step+1}] loss={float(metrics['loss']):.4f} "
                      f"pol={float(metrics['policy_loss']):.4f} val={float(metrics['value_loss']):.4f} "
                      f"mae={float(metrics['value_mae']):.3f}")

        # (Optional) evaluate vs. random or a previous snapshot here

    return params
# Batched apply for many positions at once (roots or leaves)
@jax.jit
def model_forward_batched(params, model, feats_b: jnp.ndarray, legal_masks_b: jnp.ndarray):
    """
    Efficiently evaluate multiple game positions in parallel.
    
    This function processes a batch of game states simultaneously, applying
    the neural network and masking illegal moves. Used for efficient MCTS
    leaf evaluation.
    
    Args:
        params: Model parameters
        model: Neural network model (TTTNet)
        feats_b: Batch of game state features (B, 3, 3, 3)
        legal_masks_b: Batch of legal move masks (B, 9)
        
    Returns:
        Tuple of (move_probabilities, values):
        - move_probabilities: (B, 9) probability distributions over moves
        - values: (B,) predicted game outcomes
        
    Example:
        >>> # Evaluate 32 positions at once
        >>> feats_batch = jnp.zeros((32, 3, 3, 3))
        >>> masks_batch = jnp.ones((32, 9))  # All moves legal
        >>> probs, values = model_forward_batched(params, model, feats_batch, masks_batch)
        >>> print(f"Processed {probs.shape[0]} positions")
        >>> print(f"Average value: {values.mean():.3f}")
    """
    logits, value = model.apply(params, feats_b)            # (B,9),(B,)
    # Mask+softmax per item
    def _per_item(l, m):
        masked = jnp.where(m > 0.5, l, -1e9)
        return jax.nn.softmax(masked)
    probs = jax.vmap(_per_item)(logits, legal_masks_b)      # (B,9)
    return probs, value

def eval_many_positions(model, params, envs: List[TicTacToe]):
    """
    Evaluate multiple TicTacToe game environments using the neural network.
    
    This function takes a list of game environments, converts them to features,
    creates legal move masks, and evaluates them using the batched forward pass.
    Returns policy and value estimates for each position.
    
    Args:
        model: Neural network model (TTTNet)
        params: Model parameters
        envs: List of TicTacToe game environments
        
    Returns:
        List of (priors_dict, value) tuples:
        - priors_dict: Dictionary mapping legal actions to probabilities
        - value: Predicted game outcome from current player's perspective
        
    Example:
        >>> # Create some game positions
        >>> envs = [TicTacToe() for _ in range(5)]
        >>> # Play some moves in each
        >>> for i, env in enumerate(envs):
        ...     env = env.step(i % 9)  # Different first moves
        >>> 
        >>> # Evaluate all positions
        >>> results = eval_many_positions(model, params, envs)
        >>> for i, (priors, value) in enumerate(results):
        ...     print(f"Position {i}: value={value:.3f}, best_move={max(priors, key=priors.get)}")
    """
    feats = np.stack([make_features(e.board, e.to_move) for e in envs], axis=0)
    masks = []
    for e in envs:
        legal = e.legal_actions()
        m = np.zeros(9, dtype=np.float32); m[legal] = 1.0
        masks.append(m)
    masks = np.stack(masks, 0)
    probs_b, values_b = model_forward_batched(params, model, jnp.asarray(feats), jnp.asarray(masks))
    # Convert to Python types
    probs_b = np.array(probs_b); values_b = np.array(values_b)
    # Return list of (priors_dict, value)
    out = []
    for i, env in enumerate(envs):
        legal = env.legal_actions()
        priors = {int(a): float(probs_b[i, a]) for a in legal}
        s = sum(priors.values()) or 1.0
        for a in list(priors.keys()): priors[a] /= s
        out.append((priors, float(values_b[i])))
    return out
    
def mcts_search_with_leaf_batching(root_env: TicTacToe, mcts: PUCT_MCTS,
                                   model: TTTNet, params, n_simulations: int = 512, batch_size: int = 64,
                                   temperature: float = 1.0):
    """
    Perform MCTS search with efficient batched leaf evaluation.
    
    This function implements MCTS with neural network guidance, but optimizes
    performance by batching leaf evaluations. Instead of evaluating each leaf
    individually, it collects multiple leaves and evaluates them together.
    
    The algorithm:
    1. Perform MCTS selection until reaching unexpanded nodes
    2. Collect multiple leaves for batch evaluation
    3. Evaluate all leaves simultaneously using the neural network
    4. Expand nodes and backup values
    5. Repeat until reaching simulation limit
    6. Select move based on visit counts
    
    Args:
        root_env: Starting game environment
        mcts: MCTS instance with policy-value function
        model: Neural network model (TTTNet)
        params: Model parameters
        n_simulations: Total number of MCTS simulations (default: 512)
        batch_size: Number of leaves to evaluate per batch (default: 64)
        temperature: Move selection temperature (default: 1.0)
        
    Returns:
        Selected action (integer 0-8) based on visit counts
        
    Example:
        >>> # Initialize MCTS and model
        >>> mcts = PUCT_MCTS(policy_value_fn=pv_fn, c_puct=1.5)
        >>> model = TTTNet()
        >>> params = model.init(key, dummy_input)
        >>> 
        >>> # Create starting position
        >>> env = TicTacToe()
        >>> 
        >>> # Search with batching for efficiency
        >>> action = mcts_search_with_leaf_batching(
        ...     env, mcts, model, params,
        ...     n_simulations=256,
        ...     batch_size=32,
        ...     temperature=1.0
        ... )
        >>> print(f"Selected action: {action}")
        >>> print(f"Action corresponds to position: ({action//3}, {action%3})")
    """
    # Prepare root
    root = Node(key=state_to_key(root_env), to_move=root_env.to_move)
    mcts._expand(root_env, root)
    mcts._add_root_dirichlet_noise(root)
    mcts.last_root = root  # keep for visit extraction

    pending = []   # list of (path, leaf_env, parent_node_for_child, action_taken)
    done = 0

    while done < n_simulations:
        # 1) Collect up to batch_size leaves (selection+expansion only)
        batch_paths, batch_envs, batch_parents, batch_actions = [], [], [], []
        k = min(batch_size, n_simulations - done)
        for _ in range(k):
            node = root
            env = root_env.clone()
            path = []
            # Descend until we hit an unexpanded child or terminal
            while True:
                if env.is_terminal():
                    # Terminal: back up exact outcome immediately
                    v = env.result_from(root_env.to_move)
                    mcts._backup(path, leaf_value=v)
                    break
                a = mcts._puct_select(node)
                path.append((node, a))
                if a not in node.children:
                    # Expansion deferred — gather for batch eval
                    env_next = env.step(a)
                    batch_paths.append(path)
                    batch_envs.append(env_next)
                    batch_parents.append(node)
                    batch_actions.append(a)
                    break
                env = env.step(a)
                node = node.children[a]

        # 2) Evaluate all leaves in one shot
        if batch_envs:
            evals = eval_many_positions(model, params, batch_envs)  # list of (priors, value)
            for path, env_next, parent, a, (priors, val) in zip(batch_paths, batch_envs, batch_parents, batch_actions, evals):
                child = Node(key=state_to_key(env_next), to_move=env_next.to_move, parent=parent, parent_action=a)
                child.P = priors
                parent.children[a] = child
                mcts._backup(path, leaf_value=val, leaf_env_to_move=env_next.to_move, root_player=root_env.to_move)

        done += k

    # 3) Pick move from visit counts
    return mcts._select_action_from_visits(root, temperature)


# ============================
# Complete Usage Examples
# ============================

"""
Complete example of training a Tic-Tac-Toe agent using MCTS and neural networks.

This section demonstrates how to use all the components together to train
an AlphaZero-style agent for Tic-Tac-Toe.

Example 1: Basic Training Loop
    >>> # Import required classes (these would be defined elsewhere)
    >>> from your_game_module import TicTacToe, TTTNet, PUCT_MCTS, Node, state_to_key, make_features, make_flax_policy_value_fn
    >>> import jax
    >>> import numpy as np
    >>> 
    >>> # Set random seed for reproducibility
    >>> seed = 42
    >>> np.random.seed(seed)
    >>> key = jax.random.PRNGKey(seed)
    >>> 
    >>> # Initialize model
    >>> model = TTTNet()
    >>> dummy_input = jnp.zeros((1, 3, 3, 3))
    >>> params = model.init(key, dummy_input)
    >>> 
    >>> # Create MCTS with neural guidance
    >>> pv_fn = make_flax_policy_value_fn(model, params)
    >>> mcts = PUCT_MCTS(
    ...     policy_value_fn=pv_fn,
    ...     c_puct=1.5,
    ...     dirichlet_alpha=0.3,
    ...     root_noise_frac=0.25
    ... )
    >>> 
    >>> # Train the agent
    >>> trained_params = train_loop(
    ...     num_iters=20,
    ...     games_per_iter=8,
    ...     batch_size=32,
    ...     train_steps=50,
    ...     lr=0.001,
    ...     seed=seed
    ... )
    >>> print("Training completed!")

Example 2: Manual Self-Play and Training
    >>> # Create replay buffer
    >>> buffer = ReplayBuffer(capacity=10000)
    >>> 
    >>> # Generate training data through self-play
    >>> for game in range(10):
    ...     samples = self_play_game(mcts, model, params, temperature=1.0, temp_moves=6)
    ...     buffer.add_many(samples)
    ...     print(f"Game {game}: generated {len(samples)} samples")
    >>> 
    >>> print(f"Buffer now contains {len(buffer)} samples")
    >>> 
    >>> # Train on collected data
    >>> optimizer = make_optimizer(lr=0.001)
    >>> opt_state = optimizer.init(params)
    >>> 
    >>> for step in range(100):
    ...     feats, pis, zs = buffer.sample(batch_size=32)
    ...     params, opt_state, metrics = train_step(
    ...         params, opt_state, model,
    ...         jnp.asarray(feats), jnp.asarray(pis), jnp.asarray(zs),
    ...         optimizer
    ...     )
    ...     if step % 20 == 0:
    ...         print(f"Step {step}: loss={metrics['loss']:.4f}")

Example 3: Evaluating Game Positions
    >>> # Create some game positions to evaluate
    >>> positions = []
    >>> for i in range(5):
    ...     env = TicTacToe()
    ...     # Play some moves
    ...     for j in range(i):
    ...         env = env.step(j % 9)
    ...     positions.append(env)
    >>> 
    >>> # Evaluate all positions
    >>> results = eval_many_positions(model, params, positions)
    >>> for i, (priors, value) in enumerate(results):
    ...     best_move = max(priors, key=priors.get)
    ...     print(f"Position {i}: value={value:.3f}, best_move={best_move}")

Example 4: Using Batched MCTS Search
    >>> # Create a game position
    >>> env = TicTacToe()
    >>> env = env.step(4)  # Play center
    >>> 
    >>> # Search for best move with batching
    >>> action = mcts_search_with_leaf_batching(
    ...     env, mcts, model, params,
    ...     n_simulations=256,
    ...     batch_size=32,
    ...     temperature=1.0
    ... )
    >>> print(f"Best move: {action} (position {action//3}, {action%3})")

Example 5: Monitoring Training Progress
    >>> # Custom training loop with detailed monitoring
    >>> buffer = ReplayBuffer(capacity=50000)
    >>> optimizer = make_optimizer(lr=0.001)
    >>> opt_state = optimizer.init(params)
    >>> 
    >>> for iteration in range(10):
    ...     # Generate data
    ...     all_samples = []
    ...     for _ in range(8):
    ...         samples = self_play_game(mcts, model, params, temperature=1.0)
    ...         all_samples.extend(samples)
    ...     buffer.add_many(all_samples)
    ...     
    ...     # Train
    ...     total_loss = 0
    ...     for step in range(50):
    ...         feats, pis, zs = buffer.sample(batch_size=32)
    ...         params, opt_state, metrics = train_step(
    ...             params, opt_state, model,
    ...             jnp.asarray(feats), jnp.asarray(pis), jnp.asarray(zs),
    ...             optimizer
    ...         )
    ...         total_loss += metrics['loss']
    ...     
    ...     avg_loss = total_loss / 50
    ...     print(f"Iteration {iteration}: avg_loss={avg_loss:.4f}, buffer_size={len(buffer)}")
    ...     
    ...     # Update MCTS with new parameters
    ...     pv_fn = make_flax_policy_value_fn(model, params)
    ...     mcts.policy_value_fn = pv_fn

Key Concepts Explained:

1. Self-Play: The agent plays against itself to generate training data.
   Each game produces (state, policy, value) samples where:
   - state: Current game board representation
   - policy: Move probabilities from MCTS visit counts
   - value: Game outcome from that position's perspective

2. MCTS Integration: Monte Carlo Tree Search uses the neural network
   to guide its search. The network provides:
   - Policy priors for move selection
   - Value estimates for position evaluation

3. Training Loop: The neural network learns to predict:
   - MCTS-derived move probabilities (policy loss)
   - Actual game outcomes (value loss)

4. Batched Evaluation: Multiple positions are evaluated simultaneously
   for computational efficiency, especially important for MCTS leaf evaluation.

5. Temperature Control: High temperature early in games encourages exploration,
   while low temperature later focuses on exploitation of good moves.

Performance Tips:
- Use larger batch sizes for better GPU utilization
- Increase MCTS simulations for stronger play (but slower)
- Monitor loss curves to detect overfitting
- Save model checkpoints periodically
- Consider curriculum learning (start with simpler positions)
"""
