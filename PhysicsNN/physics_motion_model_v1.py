"""
Physics-Informed Neural Network for 1D Motion
==============================================
Models a single mass in 1D with:
  - Viscous damping:     F_d = -b * v
  - Kinetic friction:    F_k = -c * m * tanh(v / eps)   [smooth sign approximation]
  - Static friction:     emerges naturally as v -> 0

Learned parameters: b (damping), c (friction x gravity), eps (sign sharpness), dt
All other operations are fixed, interpretable physics layers.

Inputs:  x(t-1), x(t), m
Output:  x(t+1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ─────────────────────────────────────────────
# 1.  SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

@dataclass
class PhysicsParams:
    b: float = 0.8          # damping coefficient  [N·s/m]
    mu_k: float = 0.25      # kinetic friction coefficient
    mu_s: float = 0.30      # static friction coefficient
    g: float = 9.81         # gravitational acceleration [m/s²]
    dt: float = 0.02        # time step [s]
    v_static_thresh: float = 0.01  # velocity threshold for static friction [m/s]


def simulate_trajectory(
    x0: float,
    v0: float,
    mass: float,
    n_steps: int,
    params: PhysicsParams,
    noise_std: float = 0.001,
) -> np.ndarray:
    """
    Simulate ground-truth 1D motion using semi-implicit Euler integration.
    Returns array of positions, shape (n_steps + 1,).
    """
    positions = np.zeros(n_steps + 1)
    positions[0] = x0

    v = v0
    dt = params.dt

    for i in range(n_steps):
        if abs(v) < params.v_static_thresh:
            # Static friction: object is at rest, no external force -> stays put
            v = 0.0
            positions[i + 1] = positions[i] + np.random.normal(0, noise_std)
        else:
            # Kinetic friction + damping
            f_damping = -params.b * v
            f_friction = -params.mu_k * mass * params.g * np.sign(v)
            a = (f_damping + f_friction) / mass

            # Semi-implicit Euler: update velocity first, then position
            v_new = v + a * dt

            # Clamp: friction cannot reverse velocity direction
            if np.sign(v_new) != np.sign(v):
                v_new = 0.0

            v = v_new
            positions[i + 1] = positions[i] + v * dt + np.random.normal(0, noise_std)

    return positions


def generate_dataset(
    n_trajectories: int = 500,
    n_steps: int = 100,
    params: PhysicsParams = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate dataset of (x_prev, x_curr, mass) -> x_next triplets.

    Returns:
        X : (N, 3)  — [x(t-1), x(t), mass]
        y : (N, 1)  — [x(t+1)]
    """
    if params is None:
        params = PhysicsParams()

    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for _ in range(n_trajectories):
        mass = rng.uniform(0.5, 5.0)          # kg
        x0   = rng.uniform(-10.0, 10.0)      # m
        v0   = rng.uniform(-8.0, 8.0)        # m/s

        pos = simulate_trajectory(x0, v0, mass, n_steps, params)

        for t in range(1, n_steps):
            X_list.append([pos[t - 1], pos[t], mass])
            y_list.append([pos[t + 1]])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    return X, y


# ─────────────────────────────────────────────
# 2.  MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class DifferentiatorLayer(nn.Module):
    """
    Finite-difference first-order differentiator.
    Estimates velocity from two consecutive positions.
    dt is passed in (not owned here) so the learnable dt is centralised.
    """
    def forward(self, x_prev: torch.Tensor, x_curr: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return (x_curr - x_prev) / dt


class ForceEstimatorLayer(nn.Module):
    """
    Structured force estimator encoding the physical form of the forces:

        F = -b * v  -  c * m * tanh(v / eps)

    where:
        b   : viscous damping coefficient          (learned)
        c   : effective friction constant μ_k * g  (learned)
        eps : smoothness of the sign function       (learned)

    All parameters are constrained positive via softplus.
    """
    def __init__(self):
        super().__init__()
        # Initialise in log-space so raw params are unconstrained reals
        self._log_b   = nn.Parameter(torch.tensor(-1.0))   # exp(-1) ≈ 0.37
        self._log_c   = nn.Parameter(torch.tensor(-1.0))
        self._log_eps = nn.Parameter(torch.tensor(-2.0))   # exp(-2) ≈ 0.14

    @property
    def b(self) -> torch.Tensor:
        return torch.exp(self._log_b)

    @property
    def c(self) -> torch.Tensor:
        return torch.exp(self._log_c)

    @property
    def eps(self) -> torch.Tensor:
        return torch.exp(self._log_eps)

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        f_damping  = -self.b * v
        f_friction = -self.c * m * torch.tanh(v / (self.eps + 1e-8))
        return f_damping + f_friction


class IntegratorLayer(nn.Module):
    """
    First-order (Euler) integrator:  y(t+1) = y(t) + dy/dt * dt
    """
    def forward(self, y: torch.Tensor, dy_dt: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return y + dy_dt * dt


class PhysicsMotionModel(nn.Module):
    """
    Full physics-informed model for 1D motion.

    Forward pass:
        1. Differentiator  : v(t)    = [x(t) - x(t-1)] / dt
        2. Force estimator : F       = -b·v - c·m·tanh(v/ε)
        3. Newton's 2nd    : a(t)    = F / m
        4. Integrator      : v(t+1)  = v(t) + a(t)·dt
        5. Integrator      : x(t+1)  = x(t) + v(t+1)·dt
    """
    def __init__(self, dt_init: float = 0.05):
        super().__init__()
        # Learned time step (stored as log so it stays positive)
        self._log_dt = nn.Parameter(torch.tensor(np.log(dt_init)))

        self.differentiator   = DifferentiatorLayer()
        self.force_estimator  = ForceEstimatorLayer()
        self.integrator       = IntegratorLayer()

    @property
    def dt(self) -> torch.Tensor:
        return torch.exp(self._log_dt)

    def forward(self, x_prev: torch.Tensor, x_curr: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        dt = self.dt

        # 1. Velocity estimate
        v = self.differentiator(x_prev, x_curr, dt)

        # 2. Net force
        F = self.force_estimator(v, m)

        # 3. Acceleration
        a = F / m

        # 4. Update velocity
        v_next = self.integrator(v, a, dt)

        # 5. Update position
        x_next = self.integrator(x_curr, v_next, dt)

        return x_next

    def summary(self) -> dict:
        return {
            "dt":  self.dt.item(),
            "b":   self.force_estimator.b.item(),
            "c":   self.force_estimator.c.item(),
            "eps": self.force_estimator.eps.item(),
        }


# ─────────────────────────────────────────────
# 3.  TRAINING
# ─────────────────────────────────────────────

def train(
    model: PhysicsMotionModel,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-2,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn   = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    n = X_train.shape[0]

    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            xb = X_train[batch_idx]
            yb = y_train[batch_idx]

            x_prev = xb[:, 0:1]
            x_curr = xb[:, 1:2]
            mass   = xb[:, 2:3]

            pred = model(x_prev, x_curr, mass)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val[:, 0:1], X_val[:, 1:2], X_val[:, 2:3])
            val_loss = loss_fn(val_pred, y_val).item()

        avg_train = epoch_loss / n_batches
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 50 == 0:
            s = model.summary()
            print(
                f"Epoch {epoch+1:4d} | "
                f"train={avg_train:.6f}  val={val_loss:.6f} | "
                f"dt={s['dt']:.4f}  b={s['b']:.4f}  c={s['c']:.4f}  eps={s['eps']:.4f}"
            )

    return history


# ─────────────────────────────────────────────
# 4.  EVALUATION & PLOTTING
# ─────────────────────────────────────────────

def rollout(
    model: PhysicsMotionModel,
    x0: float,
    v0: float,
    mass: float,
    n_steps: int,
    params: PhysicsParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare model rollout against ground-truth simulation.
    Returns (gt_positions, pred_positions).
    """
    gt = simulate_trajectory(x0, v0, mass, n_steps, params, noise_std=0.0)

    model.eval()
    pred = [x0]
    # Seed with first two ground-truth positions (one step of history needed)
    x_prev = torch.tensor([[gt[0]]], dtype=torch.float32)
    x_curr = torch.tensor([[gt[1]]], dtype=torch.float32)
    m_t    = torch.tensor([[mass]],  dtype=torch.float32)
    pred.append(gt[1])

    with torch.no_grad():
        for _ in range(n_steps - 1):
            x_next = model(x_prev, x_curr, m_t)
            pred.append(x_next.item())
            x_prev = x_curr
            x_curr = x_next

    return gt, np.array(pred)


def plot_results(history: dict, gt: np.ndarray, pred: np.ndarray, params: PhysicsParams):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Physics-Informed Motion Model — 1D Friction & Damping", fontsize=13, fontweight="bold")

    # Loss curves
    ax = axes[0]
    ax.semilogy(history["train_loss"], label="Train MSE", color="#2563eb")
    ax.semilogy(history["val_loss"],   label="Val MSE",   color="#dc2626", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trajectory comparison
    ax = axes[1]
    t = np.arange(len(gt)) * params.dt
    ax.plot(t, gt,   label="Ground Truth",    color="#16a34a", linewidth=2)
    ax.plot(t, pred, label="Model Rollout",   color="#7c3aed", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Trajectory Rollout")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    TRUE_PARAMS = PhysicsParams(b=0.8, mu_k=0.25, mu_s=0.30, g=9.81, dt=0.02)

    print("Generating dataset...")
    X, y = generate_dataset(n_trajectories=500, n_steps=100, params=TRUE_PARAMS)

    # Train / val split
    n_total = X.shape[0]
    n_train = int(0.85 * n_total)
    perm    = torch.randperm(n_total)
    X_train, y_train = X[perm[:n_train]],  y[perm[:n_train]]
    X_val,   y_val   = X[perm[n_train:]], y[perm[n_train:]]
    print(f"Dataset: {n_train} train, {n_total - n_train} val samples")

    print("\nTraining model...")
    # Initialise dt deliberately off from truth to verify learning
    model = PhysicsMotionModel(dt_init=0.05)
    history = train(model, X_train, y_train, X_val, y_val, n_epochs=300, lr=1e-2)

    print("\n── Learned parameters ──────────────────────")
    s = model.summary()
    print(f"  dt  = {s['dt']:.4f}   (true: {TRUE_PARAMS.dt:.4f})")
    print(f"  b   = {s['b']:.4f}   (true: {TRUE_PARAMS.b:.4f})")
    print(f"  c   = {s['c']:.4f}   (true: μ_k·g = {TRUE_PARAMS.mu_k * TRUE_PARAMS.g:.4f})")
    print(f"  eps = {s['eps']:.4f}")
    print("────────────────────────────────────────────\n")

    # Rollout test on a fresh trajectory
    gt, pred = rollout(model, x0=8.0, v0=-6.0, mass=2.0, n_steps=150, params=TRUE_PARAMS)
    mae = np.mean(np.abs(gt - pred))
    print(f"Rollout MAE over 150 steps: {mae:.5f} m")

    plot_results(history, gt, pred, TRUE_PARAMS)
