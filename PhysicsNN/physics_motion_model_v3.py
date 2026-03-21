"""
Physics-Informed Neural Network for 1D Motion  —  v3
=====================================================
Adds an explicit external force input F_ext to v2.

Force decomposition:
    F_total = F_internal(v, m)  +  F_ext

    F_internal  — learned by the antisymmetric MLP (damping, friction)
    F_ext       — raw observable input, passed directly to Newton's layer

Architecture:

    x(t-1), x(t)  ──► [Differentiator]     ──► v(t)
                                                  │
    v(t), m(t)    ──► [MLP: internal force] ──► F_int(t)
                                                  │
    F_int + F_ext ──► [Sum]                 ──► F_total(t)
                                                  │
    F_total, m    ──► [Newton layer]        ──► a(t) = F_total / m
                                                  │
    v(t), a(t)    ──► [Integrator]          ──► v(t+1)
                                                  │
    x(t), v(t+1)  ──► [Integrator]         ──► x(t+1)

Input features:  [x(t-1), x(t), mass, F_ext(t)]
Target label:    x(t+1)

External force profiles used during training:
    - Zero (passive decay)
    - Constant (steady push)
    - Step (sudden change)
    - Sinusoidal (oscillating drive)
    - Random walk (noisy drive)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal


# ─────────────────────────────────────────────
# 1.  PHYSICS PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class PhysicsParams:
    b: float = 0.8
    mu_k: float = 0.25
    mu_s: float = 0.30
    g: float = 9.81
    dt: float = 0.02
    v_static_thresh: float = 0.01


# ─────────────────────────────────────────────
# 2.  EXTERNAL FORCE PROFILES
# ─────────────────────────────────────────────

def make_force_profile(
    profile: str,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a time series of external force values F_ext(t).

    Profiles:
        zero        — no external force
        constant    — fixed force throughout
        step        — force switches on/off at random times
        sinusoidal  — oscillating drive
        random_walk — Gaussian random walk (low-frequency noise)
    """
    t = np.arange(n_steps) * dt

    if profile == "zero":
        return np.zeros(n_steps)

    elif profile == "constant":
        amp = rng.uniform(-15.0, 15.0)
        return np.full(n_steps, amp)

    elif profile == "step":
        F = np.zeros(n_steps)
        n_switches = rng.integers(1, 5)
        switch_times = np.sort(rng.integers(0, n_steps, size=n_switches))
        amp = rng.uniform(-15.0, 15.0)
        for i, sw in enumerate(switch_times):
            F[sw:] = amp * ((-1) ** i)
        return F

    elif profile == "sinusoidal":
        amp   = rng.uniform(2.0, 12.0)
        freq  = rng.uniform(0.5, 3.0)   # Hz
        phase = rng.uniform(0, 2 * np.pi)
        return amp * np.sin(2 * np.pi * freq * t + phase)

    elif profile == "random_walk":
        steps = rng.normal(0, 0.5, size=n_steps)
        F = np.cumsum(steps)
        F = np.clip(F, -15.0, 15.0)
        return F

    else:
        raise ValueError(f"Unknown profile: {profile}")


PROFILES = ["zero", "constant", "step", "sinusoidal", "random_walk"]


# ─────────────────────────────────────────────
# 3.  SIMULATION
# ─────────────────────────────────────────────

def simulate_trajectory(
    x0: float,
    v0: float,
    mass: float,
    F_ext_series: np.ndarray,
    params: PhysicsParams,
    noise_std: float = 0.001,
) -> np.ndarray:
    """
    Simulate ground-truth 1D motion with an external force time series.
    Returns positions of shape (n_steps + 1,).
    """
    n_steps   = len(F_ext_series)
    positions = np.zeros(n_steps + 1)
    positions[0] = x0
    v = v0
    dt = params.dt

    for i in range(n_steps):
        F_ext = F_ext_series[i]
        net_force_threshold = params.mu_s * mass * params.g

        if abs(v) < params.v_static_thresh:
            # Static friction: object moves only if external force exceeds static limit
            if abs(F_ext) > net_force_threshold:
                v = 0.0  # barely breaking free; velocity update happens below
            else:
                v = 0.0
                positions[i + 1] = positions[i] + np.random.normal(0, noise_std)
                continue

        f_internal = -params.b * v - params.mu_k * mass * params.g * np.sign(v)
        a = (f_internal + F_ext) / mass

        v_new = v + a * dt

        # Clamp: if object was moving and now would reverse, stop it first
        if abs(v) >= params.v_static_thresh and np.sign(v_new) != np.sign(v):
            if abs(F_ext) <= net_force_threshold:
                v_new = 0.0

        v = v_new
        positions[i + 1] = positions[i] + v * dt + np.random.normal(0, noise_std)

    return positions


def generate_dataset(
    n_trajectories: int = 600,
    n_steps: int = 100,
    params: PhysicsParams = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate dataset of (x_prev, x_curr, mass, F_ext) -> x_next.

    Returns:
        X : (N, 4)  — [x(t-1), x(t), mass, F_ext(t)]
        y : (N, 1)  — [x(t+1)]
    """
    if params is None:
        params = PhysicsParams()

    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    # Distribute trajectories evenly across force profiles
    per_profile = n_trajectories // len(PROFILES)

    for profile in PROFILES:
        for _ in range(per_profile):
            mass  = rng.uniform(0.5, 5.0)
            x0    = rng.uniform(-10.0, 10.0)
            v0    = rng.uniform(-8.0, 8.0)
            F_ext = make_force_profile(profile, n_steps, params.dt, rng)
            pos   = simulate_trajectory(x0, v0, mass, F_ext, params)

            for t in range(1, n_steps):
                X_list.append([pos[t - 1], pos[t], mass, F_ext[t]])
                y_list.append([pos[t + 1]])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    return X, y


# ─────────────────────────────────────────────
# 4.  PHYSICS LAYERS  (fixed, not learned)
# ─────────────────────────────────────────────

class DifferentiatorLayer(nn.Module):
    """v(t) = [x(t) - x(t-1)] / dt"""
    def forward(self, x_prev, x_curr, dt):
        return (x_curr - x_prev) / dt


class NewtonLayer(nn.Module):
    """a = F_total / m"""
    def forward(self, F_total, m):
        return F_total / m


class IntegratorLayer(nn.Module):
    """y(t+1) = y(t) + dy_dt * dt"""
    def forward(self, y, dy_dt, dt):
        return y + dy_dt * dt


# ─────────────────────────────────────────────
# 5.  MLP INTERNAL FORCE ESTIMATOR
# ─────────────────────────────────────────────

class MLPForceEstimator(nn.Module):
    """
    Antisymmetric MLP for internal forces (damping + friction).

    Inputs:  v(t), m
    Output:  F_internal  (scalar per sample)

    Antisymmetry enforced:
        F(v, m) = [net(+v, m) - net(-v, m)] / 2
    => F(-v, m) = -F(v, m)  always holds exactly.

    F_ext is NOT handled here — it bypasses this layer entirely.
    """

    def __init__(self, hidden_dim: int = 32, n_hidden: int = 2):
        super().__init__()

        layers = []
        in_dim = 2  # [v, m]
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        inp_pos = torch.cat([+v, m], dim=-1)
        inp_neg = torch.cat([-v, m], dim=-1)
        return (self.net(inp_pos) - self.net(inp_neg)) / 2.0

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────
# 6.  FULL MODEL
# ─────────────────────────────────────────────

class PhysicsMotionModel(nn.Module):
    """
    Physics-informed 1D motion model with external force support.

    Input:  x_prev, x_curr, mass, F_ext  (all shape [B, 1])
    Output: x_next                        (shape [B, 1])

    Force budget:
        F_internal  = MLP(v, m)         — learned: damping + friction
        F_total     = F_internal + F_ext — external force added directly
        a           = F_total / m        — Newton's 2nd law
    """

    def __init__(self, dt_init: float = 0.05, hidden_dim: int = 32, n_hidden: int = 2):
        super().__init__()
        self._log_dt = nn.Parameter(torch.tensor(np.log(dt_init)))

        # Fixed physics layers
        self.differentiator = DifferentiatorLayer()
        self.newton         = NewtonLayer()
        self.integrator     = IntegratorLayer()

        # Learned internal force estimator
        self.force_estimator = MLPForceEstimator(hidden_dim=hidden_dim, n_hidden=n_hidden)

    @property
    def dt(self) -> torch.Tensor:
        return torch.exp(self._log_dt)

    def forward(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        m:      torch.Tensor,
        F_ext:  torch.Tensor,
    ) -> torch.Tensor:
        dt = self.dt

        # 1. Estimate velocity
        v = self.differentiator(x_prev, x_curr, dt)

        # 2. Internal force (MLP: damping + friction)
        F_int = self.force_estimator(v, m)

        # 3. Total force — external force added directly, no learning needed
        F_total = F_int + F_ext

        # 4. Acceleration
        a = self.newton(F_total, m)

        # 5–6. Integrate
        v_next = self.integrator(v, a, dt)
        x_next = self.integrator(x_curr, v_next, dt)

        return x_next

    def summary(self) -> dict:
        return {
            "dt":       self.dt.item(),
            "n_params": self.force_estimator.n_params(),
        }


# ─────────────────────────────────────────────
# 7.  TRAINING
# ─────────────────────────────────────────────

def train(
    model:    PhysicsMotionModel,
    X_train:  torch.Tensor,
    y_train:  torch.Tensor,
    X_val:    torch.Tensor,
    y_val:    torch.Tensor,
    n_epochs: int = 300,
    batch_size: int = 256,
    lr:       float = 1e-3,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn   = nn.MSELoss()
    history   = {"train_loss": [], "val_loss": []}
    n         = X_train.shape[0]

    for epoch in range(n_epochs):
        model.train()
        idx        = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            xb, yb = X_train[batch], y_train[batch]

            pred = model(xb[:, 0:1], xb[:, 1:2], xb[:, 2:3], xb[:, 3:4])
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
            val_pred = model(X_val[:, 0:1], X_val[:, 1:2], X_val[:, 2:3], X_val[:, 3:4])
            val_loss = loss_fn(val_pred, y_val).item()

        history["train_loss"].append(epoch_loss / n_batches)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 50 == 0:
            s = model.summary()
            print(
                f"Epoch {epoch+1:4d} | "
                f"train={history['train_loss'][-1]:.6f}  val={val_loss:.6f} | "
                f"dt={s['dt']:.4f}"
            )

    return history


# ─────────────────────────────────────────────
# 8.  ROLLOUT & EVALUATION
# ─────────────────────────────────────────────

def rollout(
    model:   PhysicsMotionModel,
    x0:      float,
    v0:      float,
    mass:    float,
    F_ext_series: np.ndarray,
    params:  PhysicsParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Autoregressive rollout vs. ground-truth simulation."""
    n_steps = len(F_ext_series)
    gt      = simulate_trajectory(x0, v0, mass, F_ext_series, params, noise_std=0.0)

    model.eval()
    pred   = [gt[0], gt[1]]
    x_prev = torch.tensor([[gt[0]]], dtype=torch.float32)
    x_curr = torch.tensor([[gt[1]]], dtype=torch.float32)
    m_t    = torch.tensor([[mass]],  dtype=torch.float32)

    with torch.no_grad():
        for i in range(n_steps - 1):
            F = torch.tensor([[F_ext_series[i + 1]]], dtype=torch.float32)
            x_next = model(x_prev, x_curr, m_t, F)
            pred.append(x_next.item())
            x_prev = x_curr
            x_curr = x_next

    return gt, np.array(pred)


def plot_results(history: dict, rollout_data: dict, params: PhysicsParams):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        "Physics-Informed Motion Model v3 — With External Force",
        fontsize=13, fontweight="bold"
    )

    # ── Loss curves ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.semilogy(history["train_loss"], label="Train MSE", color="#2563eb", linewidth=1.5)
    ax.semilogy(history["val_loss"],   label="Val MSE",   color="#dc2626", linewidth=1.5, linestyle="--")
    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Learned internal force profile ───────────────────────────────────────
    ax = axes[0, 1]
    v_range = torch.linspace(-8, 8, 200).unsqueeze(1)
    m_fixed = torch.full_like(v_range, 2.0)
    true_F  = (
        -params.b * v_range.numpy()
        - params.mu_k * 2.0 * params.g * np.sign(v_range.numpy())
    )

    ax.plot(v_range.numpy(), true_F, color="#16a34a", linewidth=2.5, label="True F_internal (m=2)")

    model = rollout_data[list(rollout_data.keys())[0]]["model"]
    model.eval()
    with torch.no_grad():
        F_pred = model.force_estimator(v_range, m_fixed).numpy()
    ax.plot(v_range.numpy(), F_pred, color="#7c3aed", linewidth=1.5,
            linestyle="--", label="Learned F_internal")

    ax.set_title("Learned Internal Force  (m = 2 kg)")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Force [N]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Trajectory rollouts by profile ───────────────────────────────────────
    profile_colors = {
        "zero":       "#64748b",
        "constant":   "#2563eb",
        "step":       "#dc2626",
        "sinusoidal": "#d97706",
        "random_walk":"#7c3aed",
    }

    plot_profiles = ["zero", "sinusoidal", "step", "random_walk"]
    axes_flat = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]

    for ax, profile in zip(axes_flat, plot_profiles):
        data = rollout_data[profile]
        n    = len(data["gt"])
        t    = np.arange(n) * params.dt
        mae  = np.mean(np.abs(data["gt"] - data["pred"]))
        color = profile_colors[profile]

        ax.plot(t, data["gt"],   color="#16a34a",  linewidth=2,   label="Ground Truth")
        ax.plot(t, data["pred"], color=color,       linewidth=1.5, linestyle="--", label="Model Rollout")

        # F_ext has n_steps points; t has n_steps+1 — trim t to match
        t_fext = t[:len(data["F_ext"])]
        ax2 = ax.twinx()
        ax2.fill_between(t_fext, data["F_ext"], alpha=0.12, color=color)
        ax2.plot(t_fext, data["F_ext"], color=color, linewidth=0.8, alpha=0.5)
        ax2.set_ylabel("F_ext [N]", fontsize=7, color=color)
        ax2.tick_params(axis="y", labelsize=7, colors=color)

        ax.set_title(f"{profile.replace('_', ' ').title()}  —  MAE {mae:.4f} m")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    TRUE_PARAMS = PhysicsParams(b=0.8, mu_k=0.25, mu_s=0.30, g=9.81, dt=0.02)

    print("Generating dataset...")
    X, y = generate_dataset(n_trajectories=600, n_steps=100, params=TRUE_PARAMS)

    n_total = X.shape[0]
    n_train = int(0.85 * n_total)
    perm    = torch.randperm(n_total)
    X_train, y_train = X[perm[:n_train]],  y[perm[:n_train]]
    X_val,   y_val   = X[perm[n_train:]], y[perm[n_train:]]
    print(f"Dataset: {n_train} train, {n_total - n_train} val\n")

    print("Training model...")
    model = PhysicsMotionModel(dt_init=0.05, hidden_dim=32, n_hidden=2)
    history = train(model, X_train, y_train, X_val, y_val, n_epochs=300, lr=1e-3)

    print(f"\nLearned dt: {model.dt.item():.4f}  (true: {TRUE_PARAMS.dt:.4f})")

    # Rollout evaluation across all force profiles
    rng = np.random.default_rng(99)
    rollout_data = {}

    print("\nRollout MAE by profile:")
    for profile in PROFILES:
        F_ext = make_force_profile(profile, 150, TRUE_PARAMS.dt, rng)
        gt, pred = rollout(
            model, x0=6.0, v0=-4.0, mass=2.0,
            F_ext_series=F_ext, params=TRUE_PARAMS
        )
        mae = np.mean(np.abs(gt - pred))
        print(f"  {profile:<15} MAE = {mae:.5f} m")
        rollout_data[profile] = {"model": model, "gt": gt, "pred": pred, "F_ext": F_ext}

    plot_results(history, rollout_data, TRUE_PARAMS)
