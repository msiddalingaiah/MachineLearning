"""
Physics-Informed Neural Network for 1D Motion  —  v2
=====================================================
Evolution from v1: the structured scalar force estimator is replaced by
a small MLP, while the differentiator and integrator layers remain fixed
interpretable physics operations.

Architecture:

    x(t-1), x(t)  ──► [Differentiator] ──► v(t)
                                              │
    v(t), m(t)    ──► [MLP Force Net]  ──► F(t)
                                              │
    F(t), m(t)    ──► [Newton layer]   ──► a(t)  =  F / m
                                              │
    v(t), a(t)    ──► [Integrator]     ──► v(t+1) = v + a·dt
                                              │
    x(t), v(t+1)  ──► [Integrator]     ──► x(t+1) = x + v(t+1)·dt

Physical symmetry enforced in the MLP:
    F(-v, m) = -F(v, m)   (force reverses with velocity direction)
    Achieved via antisymmetric forward pass:
        F(v, m) = [mlp(+v, m) - mlp(-v, m)] / 2

Learned parameters:
    - MLP weights & biases
    - dt  (shared scalar, stored as log)

Fixed operations:
    - Differentiator layer
    - Newton's 2nd law division
    - Two integrator layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal


# ─────────────────────────────────────────────
# 1.  SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

@dataclass
class PhysicsParams:
    b: float = 0.8
    mu_k: float = 0.25
    mu_s: float = 0.30
    g: float = 9.81
    dt: float = 0.02
    v_static_thresh: float = 0.01


def simulate_trajectory(
    x0: float,
    v0: float,
    mass: float,
    n_steps: int,
    params: PhysicsParams,
    noise_std: float = 0.001,
) -> np.ndarray:
    positions = np.zeros(n_steps + 1)
    positions[0] = x0
    v = v0
    dt = params.dt

    for i in range(n_steps):
        if abs(v) < params.v_static_thresh:
            v = 0.0
            positions[i + 1] = positions[i] + np.random.normal(0, noise_std)
        else:
            f_damping  = -params.b * v
            f_friction = -params.mu_k * mass * params.g * np.sign(v)
            a = (f_damping + f_friction) / mass

            v_new = v + a * dt
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
) -> tuple[torch.Tensor, torch.Tensor]:
    if params is None:
        params = PhysicsParams()

    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for _ in range(n_trajectories):
        mass = rng.uniform(0.5, 5.0)
        x0   = rng.uniform(-10.0, 10.0)
        v0   = rng.uniform(-8.0, 8.0)
        pos  = simulate_trajectory(x0, v0, mass, n_steps, params)

        for t in range(1, n_steps):
            X_list.append([pos[t - 1], pos[t], mass])
            y_list.append([pos[t + 1]])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    return X, y


# ─────────────────────────────────────────────
# 2.  PHYSICS LAYERS  (fixed, not learned)
# ─────────────────────────────────────────────

class DifferentiatorLayer(nn.Module):
    """v(t) = [x(t) - x(t-1)] / dt"""
    def forward(self, x_prev: torch.Tensor, x_curr: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return (x_curr - x_prev) / dt


class NewtonLayer(nn.Module):
    """a = F / m"""
    def forward(self, F: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return F / m


class IntegratorLayer(nn.Module):
    """y(t+1) = y(t) + dy_dt * dt"""
    def forward(self, y: torch.Tensor, dy_dt: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return y + dy_dt * dt


# ─────────────────────────────────────────────
# 3.  FORCE ESTIMATORS
# ─────────────────────────────────────────────

class ScalarForceEstimator(nn.Module):
    """
    v1 baseline: structured 3-scalar estimator.
        F = -b*v - c*m*tanh(v/eps)
    """
    def __init__(self):
        super().__init__()
        self._log_b   = nn.Parameter(torch.tensor(-1.0))
        self._log_c   = nn.Parameter(torch.tensor(-1.0))
        self._log_eps = nn.Parameter(torch.tensor(-2.0))

    @property
    def b(self):   return torch.exp(self._log_b)
    @property
    def c(self):   return torch.exp(self._log_c)
    @property
    def eps(self): return torch.exp(self._log_eps)

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return -self.b * v - self.c * m * torch.tanh(v / (self.eps + 1e-8))


class MLPForceEstimator(nn.Module):
    """
    MLP-based force estimator with built-in antisymmetry.

    The network takes [v, m] as input and outputs a scalar force.

    Antisymmetry trick — enforces F(-v, m) = -F(v, m):
        F(v, m) = [net(+v, m) - net(-v, m)] / 2
    This means:
      - The model can only learn forces that reverse with velocity (physically correct)
      - Gradients flow through both forward and mirrored passes
      - No additional loss terms or constraints needed

    Architecture:  input(2) → hidden_dim → hidden_dim → output(1)
    Activations:   SiLU (smooth, captures nonlinear force profiles well)
    """

    def __init__(self, hidden_dim: int = 32, n_hidden: int = 2):
        super().__init__()

        layers = []
        in_dim = 2  # [v, m]

        for i in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Weight init: small values encourage near-linear start
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        inp_pos = torch.cat([+v, m], dim=-1)
        inp_neg = torch.cat([-v, m], dim=-1)

        # Antisymmetric combination
        F = (self.net(inp_pos) - self.net(inp_neg)) / 2.0
        return F

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────
# 4.  FULL MODEL
# ─────────────────────────────────────────────

class PhysicsMotionModel(nn.Module):
    """
    Full physics-informed 1D motion model.

    mode = 'scalar' : v1 structured estimator (3 scalars)
    mode = 'mlp'    : v2 MLP estimator (fully learned force law)

    In both cases the differentiator, Newton, and integrator layers
    are identical fixed operations — only the force estimator changes.
    """

    def __init__(
        self,
        mode: Literal["scalar", "mlp"] = "mlp",
        dt_init: float = 0.05,
        hidden_dim: int = 32,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.mode = mode
        self._log_dt = nn.Parameter(torch.tensor(np.log(dt_init)))

        # Physics layers (fixed operations, no learned params)
        self.differentiator = DifferentiatorLayer()
        self.newton         = NewtonLayer()
        self.integrator     = IntegratorLayer()

        # Force estimator (the only learned component)
        if mode == "scalar":
            self.force_estimator = ScalarForceEstimator()
        else:
            self.force_estimator = MLPForceEstimator(hidden_dim=hidden_dim, n_hidden=n_hidden)

    @property
    def dt(self) -> torch.Tensor:
        return torch.exp(self._log_dt)

    def forward(self, x_prev: torch.Tensor, x_curr: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        dt = self.dt

        v   = self.differentiator(x_prev, x_curr, dt)   # estimate velocity
        F   = self.force_estimator(v, m)                 # estimate force
        a   = self.newton(F, m)                          # a = F / m
        v1  = self.integrator(v, a, dt)                  # v(t+1)
        x1  = self.integrator(x_curr, v1, dt)            # x(t+1)

        return x1

    def summary(self) -> dict:
        s = {"mode": self.mode, "dt": self.dt.item()}
        if self.mode == "scalar":
            fe = self.force_estimator
            s.update({"b": fe.b.item(), "c": fe.c.item(), "eps": fe.eps.item()})
        else:
            s["n_params"] = self.force_estimator.n_params()
        return s


# ─────────────────────────────────────────────
# 5.  TRAINING
# ─────────────────────────────────────────────

def train(
    model: PhysicsMotionModel,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn   = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    n = X_train.shape[0]

    for epoch in range(n_epochs):
        model.train()
        idx        = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            xb, yb = X_train[batch], y_train[batch]

            pred = model(xb[:, 0:1], xb[:, 1:2], xb[:, 2:3])
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val[:, 0:1], X_val[:, 1:2], X_val[:, 2:3])
            val_loss = loss_fn(val_pred, y_val).item()

        history["train_loss"].append(epoch_loss / n_batches)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 50 == 0:
            s = model.summary()
            tag = f"dt={s['dt']:.4f}"
            if model.mode == "scalar":
                tag += f"  b={s['b']:.4f}  c={s['c']:.4f}  eps={s['eps']:.4f}"
            print(
                f"Epoch {epoch+1:4d} | "
                f"train={history['train_loss'][-1]:.6f}  val={val_loss:.6f} | {tag}"
            )

    return history


# ─────────────────────────────────────────────
# 6.  EVALUATION
# ─────────────────────────────────────────────

def rollout(
    model: PhysicsMotionModel,
    x0: float,
    v0: float,
    mass: float,
    n_steps: int,
    params: PhysicsParams,
) -> tuple[np.ndarray, np.ndarray]:
    gt = simulate_trajectory(x0, v0, mass, n_steps, params, noise_std=0.0)

    model.eval()
    pred   = [gt[0], gt[1]]
    x_prev = torch.tensor([[gt[0]]], dtype=torch.float32)
    x_curr = torch.tensor([[gt[1]]], dtype=torch.float32)
    m_t    = torch.tensor([[mass]],  dtype=torch.float32)

    with torch.no_grad():
        for _ in range(n_steps - 1):
            x_next = model(x_prev, x_curr, m_t)
            pred.append(x_next.item())
            x_prev = x_curr
            x_curr = x_next

    return gt, np.array(pred)


def plot_comparison(
    histories: dict,
    rollouts: dict,
    params: PhysicsParams,
):
    """Plot loss curves and trajectory rollouts for both models side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "Physics-Informed Motion Model — Scalar vs MLP Force Estimator",
        fontsize=13, fontweight="bold"
    )

    colors = {"scalar": "#2563eb", "mlp": "#dc2626"}
    labels = {"scalar": "Scalar (v1)", "mlp": "MLP (v2)"}

    # ── Loss curves (top row, merged) ───────────────────────────────────────
    ax = axes[0, 0]
    for mode, h in histories.items():
        ax.semilogy(h["train_loss"], label=f"{labels[mode]} — Train",
                    color=colors[mode], linewidth=1.5)
        ax.semilogy(h["val_loss"],   label=f"{labels[mode]} — Val",
                    color=colors[mode], linewidth=1.5, linestyle="--")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Learned force profile ────────────────────────────────────────────────
    ax = axes[0, 1]
    v_range = torch.linspace(-8, 8, 200).unsqueeze(1)
    m_fixed = torch.full_like(v_range, 2.0)
    true_F  = -params.b * v_range.numpy() - params.mu_k * 2.0 * params.g * np.sign(v_range.numpy())

    ax.plot(v_range.numpy(), true_F, color="#16a34a", linewidth=2.5,
            label="True force (m=2 kg)", linestyle="-")

    for mode, model in rollouts.items():
        model_obj = model["model"]
        model_obj.eval()
        with torch.no_grad():
            F_pred = model_obj.force_estimator(v_range, m_fixed).numpy()
        ax.plot(v_range.numpy(), F_pred, color=colors[mode], linewidth=1.5,
                linestyle="--", label=f"{labels[mode]} learned force")

    ax.set_title("Learned Force Profile  (m = 2 kg)")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Force [N]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Trajectory rollouts (bottom row) ────────────────────────────────────
    t = np.arange(rollouts["scalar"]["gt"].shape[0]) * params.dt

    for idx, (mode, data) in enumerate(rollouts.items()):
        ax = axes[1, idx]
        ax.plot(t, data["gt"],   color="#16a34a", linewidth=2,   label="Ground Truth")
        ax.plot(t, data["pred"], color=colors[mode], linewidth=1.5, linestyle="--",
                label=f"{labels[mode]} rollout")
        mae = np.mean(np.abs(data["gt"] - data["pred"]))
        ax.set_title(f"{labels[mode]} — MAE {mae:.4f} m")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    TRUE_PARAMS = PhysicsParams(b=0.8, mu_k=0.25, mu_s=0.30, g=9.81, dt=0.02)

    print("Generating dataset...")
    X, y = generate_dataset(n_trajectories=500, n_steps=100, params=TRUE_PARAMS)

    n_total = X.shape[0]
    n_train = int(0.85 * n_total)
    perm    = torch.randperm(n_total)
    X_train, y_train = X[perm[:n_train]],  y[perm[:n_train]]
    X_val,   y_val   = X[perm[n_train:]], y[perm[n_train:]]
    print(f"Dataset: {n_train} train, {n_total - n_train} val\n")

    histories = {}
    trained   = {}

    for mode in ["scalar", "mlp"]:
        print(f"═══ Training [{mode}] model ═══")
        model = PhysicsMotionModel(mode=mode, dt_init=0.05, hidden_dim=32, n_hidden=2)
        h = train(model, X_train, y_train, X_val, y_val, n_epochs=300, lr=1e-3)
        histories[mode] = h

        gt, pred = rollout(model, x0=8.0, v0=-6.0, mass=2.0, n_steps=150, params=TRUE_PARAMS)
        mae = np.mean(np.abs(gt - pred))
        print(f"\nRollout MAE [{mode}]: {mae:.5f} m")
        print(f"Learned dt  [{mode}]: {model.dt.item():.4f}  (true: {TRUE_PARAMS.dt:.4f})\n")

        trained[mode] = {"model": model, "gt": gt, "pred": pred}

    plot_comparison(histories, trained, TRUE_PARAMS)
