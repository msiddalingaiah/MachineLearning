"""
Physics-Informed Neural Network for 1D Motion  —  v4
=====================================================
Adds multi-step rollout loss to v3.

One-step loss (v3) only penalizes error at t+1, so the model can
accumulate drift over longer horizons without penalty. Multi-step
rollout loss unrolls K steps during training and accumulates a
weighted loss at every step, forcing the model to be globally stable.

Dataset change:
    v3: flat samples  (x_prev, x_curr, mass, F_ext_t) → x_next
    v4: sequence windows  positions[t-1 : t+K+1],  F_ext[t : t+K]
        → one sample covers K consecutive prediction steps

Training loop change:
    v3: one forward pass → one loss term
    v4: K forward passes, autoregressive (predicted x feeds next step)
        → K loss terms, weighted and summed

Loss weighting modes:
    uniform           — all steps equal weight
    linear            — weight increases linearly (penalise later steps more)
    exp_decay         — weight decreases exponentially (trust near steps more)

Curriculum warm-up (optional):
    Start with rollout_steps = 1 and ramp to K over the first
    curriculum_epochs epochs. Avoids instability early in training
    when the model is far from correct.

Model architecture (unchanged from v3):
    x(t-1), x(t)  ──► [Differentiator]     ──► v(t)
    v(t), m       ──► [MLP force]           ──► F_int
    F_int + F_ext ──► [Sum]                 ──► F_total
    F_total / m   ──► [Newton]              ──► a(t)
    v + a·dt      ──► [Integrator]          ──► v(t+1)
    x + v(t+1)·dt ──► [Integrator]         ──► x(t+1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
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

PROFILES = ["zero", "constant", "step", "sinusoidal", "random_walk"]


def make_force_profile(
    profile: str,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(n_steps) * dt

    if profile == "zero":
        return np.zeros(n_steps)
    elif profile == "constant":
        return np.full(n_steps, rng.uniform(-15.0, 15.0))
    elif profile == "step":
        F = np.zeros(n_steps)
        amp = rng.uniform(-15.0, 15.0)
        for i, sw in enumerate(np.sort(rng.integers(0, n_steps, size=rng.integers(1, 5)))):
            F[sw:] = amp * ((-1) ** i)
        return F
    elif profile == "sinusoidal":
        return rng.uniform(2.0, 12.0) * np.sin(
            2 * np.pi * rng.uniform(0.5, 3.0) * t + rng.uniform(0, 2 * np.pi)
        )
    elif profile == "random_walk":
        return np.clip(np.cumsum(rng.normal(0, 0.5, size=n_steps)), -15.0, 15.0)
    else:
        raise ValueError(f"Unknown profile: {profile}")


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
    n_steps   = len(F_ext_series)
    positions = np.zeros(n_steps + 1)
    positions[0] = x0
    v  = v0
    dt = params.dt

    for i in range(n_steps):
        F_ext = F_ext_series[i]
        threshold = params.mu_s * mass * params.g

        if abs(v) < params.v_static_thresh:
            if abs(F_ext) <= threshold:
                positions[i + 1] = positions[i] + np.random.normal(0, noise_std)
                continue
            v = 0.0

        f_int = -params.b * v - params.mu_k * mass * params.g * np.sign(v)
        a     = (f_int + F_ext) / mass
        v_new = v + a * dt

        if abs(v) >= params.v_static_thresh and np.sign(v_new) != np.sign(v):
            if abs(F_ext) <= threshold:
                v_new = 0.0

        v = v_new
        positions[i + 1] = positions[i] + v * dt + np.random.normal(0, noise_std)

    return positions


# ─────────────────────────────────────────────
# 4.  SEQUENCE DATASET
# ─────────────────────────────────────────────

def generate_dataset(
    n_trajectories: int = 600,
    n_steps: int = 100,
    rollout_steps: int = 10,
    params: PhysicsParams = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate sequence windows for multi-step rollout training.

    Each window covers K consecutive prediction steps starting at time t:
        pos_window  : positions[t-1 : t+K+1]  shape (K+2,)
        fext_window : F_ext[t : t+K]           shape (K,)
        mass        : scalar

    Returns:
        pos_seqs  : (N, K+2)   — position sequences
        fext_seqs : (N, K)     — matching F_ext sequences
        masses    : (N, 1)     — object mass
    """
    if params is None:
        params = PhysicsParams()

    K   = rollout_steps
    rng = np.random.default_rng(seed)
    pos_list, fext_list, mass_list = [], [], []

    per_profile = n_trajectories // len(PROFILES)

    for profile in PROFILES:
        for _ in range(per_profile):
            mass  = rng.uniform(0.5, 5.0)
            x0    = rng.uniform(-10.0, 10.0)
            v0    = rng.uniform(-8.0, 8.0)
            F_ext = make_force_profile(profile, n_steps, params.dt, rng)
            pos   = simulate_trajectory(x0, v0, mass, F_ext, params)

            # Slide a window of width K+2 across the trajectory
            # pos has length n_steps+1, F_ext has length n_steps
            # Window at index t (1-based): pos[t-1:t+K+1], F_ext[t:t+K]
            for t in range(1, n_steps - K + 1):
                pos_list.append(pos[t - 1 : t + K + 1])   # K+2 positions
                fext_list.append(F_ext[t : t + K])         # K forces
                mass_list.append([mass])

    pos_seqs  = torch.tensor(np.array(pos_list),  dtype=torch.float32)
    fext_seqs = torch.tensor(np.array(fext_list), dtype=torch.float32)
    masses    = torch.tensor(np.array(mass_list), dtype=torch.float32)
    return pos_seqs, fext_seqs, masses


# ─────────────────────────────────────────────
# 5.  PHYSICS LAYERS  (fixed)
# ─────────────────────────────────────────────

class DifferentiatorLayer(nn.Module):
    def forward(self, x_prev, x_curr, dt):
        return (x_curr - x_prev) / dt

class NewtonLayer(nn.Module):
    def forward(self, F_total, m):
        return F_total / m

class IntegratorLayer(nn.Module):
    def forward(self, y, dy_dt, dt):
        return y + dy_dt * dt


# ─────────────────────────────────────────────
# 6.  MLP FORCE ESTIMATOR
# ─────────────────────────────────────────────

class MLPForceEstimator(nn.Module):
    """
    Antisymmetric MLP: F(v,m) = [net(+v,m) - net(-v,m)] / 2
    Guarantees F(-v, m) = -F(v, m) for all inputs.
    """
    def __init__(self, hidden_dim: int = 32, n_hidden: int = 2):
        super().__init__()
        layers, in_dim = [], 2
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, v, m):
        pos = torch.cat([+v, m], dim=-1)
        neg = torch.cat([-v, m], dim=-1)
        return (self.net(pos) - self.net(neg)) / 2.0

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────
# 7.  FULL MODEL
# ─────────────────────────────────────────────

class PhysicsMotionModel(nn.Module):
    """Single-step forward pass. Multi-step unrolling is handled in training."""

    def __init__(self, dt_init: float = 0.05, hidden_dim: int = 32, n_hidden: int = 2):
        super().__init__()
        self._log_dt     = nn.Parameter(torch.tensor(np.log(dt_init)))
        self.differentiator  = DifferentiatorLayer()
        self.force_estimator = MLPForceEstimator(hidden_dim, n_hidden)
        self.newton          = NewtonLayer()
        self.integrator      = IntegratorLayer()

    @property
    def dt(self):
        return torch.exp(self._log_dt)

    def forward(self, x_prev, x_curr, m, F_ext):
        dt    = self.dt
        v     = self.differentiator(x_prev, x_curr, dt)
        F_int = self.force_estimator(v, m)
        a     = self.newton(F_int + F_ext, m)
        v1    = self.integrator(v, a, dt)
        x1    = self.integrator(x_curr, v1, dt)
        return x1

    def summary(self):
        return {"dt": self.dt.item(), "n_params": self.force_estimator.n_params()}


# ─────────────────────────────────────────────
# 8.  LOSS WEIGHTS
# ─────────────────────────────────────────────

def make_step_weights(
    K: int,
    mode: Literal["uniform", "linear", "exp_decay"],
    device: torch.device,
) -> torch.Tensor:
    """
    Return a weight vector of length K, normalised to sum to 1.

    uniform    : [1, 1, 1, ..., 1]           — treat all steps equally
    linear     : [1, 2, 3, ..., K]           — penalise later steps more
    exp_decay  : [γ^0, γ^1, ..., γ^(K-1)]   — trust near steps more (γ=0.85)
    """
    k = torch.arange(1, K + 1, dtype=torch.float32, device=device)
    if mode == "uniform":
        w = torch.ones(K, device=device)
    elif mode == "linear":
        w = k
    elif mode == "exp_decay":
        w = 0.85 ** (k - 1)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    return w / w.sum()


# ─────────────────────────────────────────────
# 9.  TRAINING
# ─────────────────────────────────────────────

def train(
    model:             PhysicsMotionModel,
    pos_train:         torch.Tensor,
    fext_train:        torch.Tensor,
    mass_train:        torch.Tensor,
    pos_val:           torch.Tensor,
    fext_val:          torch.Tensor,
    mass_val:          torch.Tensor,
    rollout_steps:     int   = 10,
    weight_mode:       str   = "linear",
    curriculum_epochs: int   = 50,
    n_epochs:          int   = 300,
    batch_size:        int   = 256,
    lr:                float = 1e-3,
) -> dict:
    """
    Multi-step rollout training loop.

    Curriculum warm-up:
        Epochs 0..curriculum_epochs: active_K ramps from 1 → rollout_steps
        This lets the model first learn single-step accuracy, then gradually
        extend its horizon, avoiding gradient explosions early in training.

    For each batch at horizon active_K:
        1. Unroll active_K steps autoregressively
        2. Compute MSE at each step against ground-truth positions
        3. Weight the per-step losses and sum
        4. Backprop through the entire unrolled sequence
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    device    = next(model.parameters()).device
    history   = {"train_loss": [], "val_loss": [], "active_K": []}
    n         = pos_train.shape[0]
    K_max     = rollout_steps

    for epoch in range(n_epochs):
        # ── Curriculum: ramp active_K from 1 to K_max ─────────────────────
        if curriculum_epochs > 0:
            ramp    = min(epoch / max(curriculum_epochs, 1), 1.0)
            active_K = max(1, int(round(1 + ramp * (K_max - 1))))
        else:
            active_K = K_max

        weights = make_step_weights(active_K, weight_mode, device)

        # ── Training pass ──────────────────────────────────────────────────
        model.train()
        idx        = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            batch  = idx[start : start + batch_size]
            pos_b  = pos_train[batch]       # (B, K_max+2)
            fext_b = fext_train[batch]      # (B, K_max)
            mass_b = mass_train[batch]      # (B, 1)

            # Seed the rollout with the first two positions
            x_prev = pos_b[:, 0:1]
            x_curr = pos_b[:, 1:2]

            loss = torch.tensor(0.0, device=device)

            for k in range(active_K):
                F_ext  = fext_b[:, k:k+1]
                x_next = model(x_prev, x_curr, mass_b, F_ext)

                # Ground-truth position at step k is pos_b[:, k+2]
                target     = pos_b[:, k + 2 : k + 3]
                step_loss  = ((x_next - target) ** 2).mean()
                loss       = loss + weights[k] * step_loss

                # Autoregressive: feed prediction forward (detach x_prev only)
                x_prev = x_curr.detach()
                x_curr = x_next

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # ── Validation (single-step MSE for comparability) ─────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(
                pos_val[:, 0:1], pos_val[:, 1:2], mass_val, fext_val[:, 0:1]
            )
            val_loss = ((val_pred - pos_val[:, 2:3]) ** 2).mean().item()

        history["train_loss"].append(epoch_loss / n_batches)
        history["val_loss"].append(val_loss)
        history["active_K"].append(active_K)

        if (epoch + 1) % 50 == 0:
            s = model.summary()
            print(
                f"Epoch {epoch+1:4d} | K={active_K:2d} | "
                f"train={history['train_loss'][-1]:.6f}  val={val_loss:.6f} | "
                f"dt={s['dt']:.4f}"
            )

    return history


# ─────────────────────────────────────────────
# 10.  ROLLOUT & EVALUATION
# ─────────────────────────────────────────────

def rollout(
    model:        PhysicsMotionModel,
    x0:           float,
    v0:           float,
    mass:         float,
    F_ext_series: np.ndarray,
    params:       PhysicsParams,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = len(F_ext_series)
    gt      = simulate_trajectory(x0, v0, mass, F_ext_series, params, noise_std=0.0)

    model.eval()
    pred   = [gt[0], gt[1]]
    x_prev = torch.tensor([[gt[0]]], dtype=torch.float32)
    x_curr = torch.tensor([[gt[1]]], dtype=torch.float32)
    m_t    = torch.tensor([[mass]],  dtype=torch.float32)

    with torch.no_grad():
        for i in range(n_steps - 1):
            F      = torch.tensor([[F_ext_series[i + 1]]], dtype=torch.float32)
            x_next = model(x_prev, x_curr, m_t, F)
            pred.append(x_next.item())
            x_prev = x_curr
            x_curr = x_next

    return gt, np.array(pred)


# ─────────────────────────────────────────────
# 11.  PLOTTING
# ─────────────────────────────────────────────

def plot_results(history: dict, rollout_data: dict, params: PhysicsParams):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        "Physics-Informed Motion Model v4 — Multi-Step Rollout Loss",
        fontsize=13, fontweight="bold"
    )

    # ── Loss + active_K ──────────────────────────────────────────────────────
    ax  = axes[0, 0]
    ax2 = ax.twinx()
    ax.semilogy(history["train_loss"], color="#2563eb", linewidth=1.5, label="Train MSE")
    ax.semilogy(history["val_loss"],   color="#dc2626", linewidth=1.5, linestyle="--", label="Val MSE")
    ax2.plot(history["active_K"], color="#94a3b8", linewidth=1.0, linestyle=":", alpha=0.8)
    ax2.set_ylabel("Active K (rollout steps)", fontsize=7, color="#94a3b8")
    ax2.tick_params(axis="y", labelsize=7, colors="#94a3b8")
    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Learned internal force profile ───────────────────────────────────────
    ax      = axes[0, 1]
    v_range = torch.linspace(-8, 8, 200).unsqueeze(1)
    m_fixed = torch.full_like(v_range, 2.0)
    true_F  = (
        -params.b * v_range.numpy()
        - params.mu_k * 2.0 * params.g * np.sign(v_range.numpy())
    )
    ax.plot(v_range.numpy(), true_F, color="#16a34a", linewidth=2.5, label="True F_int (m=2)")

    model = next(iter(rollout_data.values()))["model"]
    model.eval()
    with torch.no_grad():
        F_pred = model.force_estimator(v_range, m_fixed).numpy()
    ax.plot(v_range.numpy(), F_pred, color="#7c3aed", linewidth=1.5,
            linestyle="--", label="Learned F_int")
    ax.set_title("Learned Internal Force  (m = 2 kg)")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Force [N]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Trajectory rollouts ───────────────────────────────────────────────────
    profile_colors = {
        "zero": "#64748b", "constant": "#2563eb",
        "step": "#dc2626", "sinusoidal": "#d97706", "random_walk": "#7c3aed",
    }
    plot_profiles = ["zero", "sinusoidal", "step", "random_walk"]
    axes_flat     = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]

    for ax, profile in zip(axes_flat, plot_profiles):
        data  = rollout_data[profile]
        n     = len(data["gt"])
        t     = np.arange(n) * params.dt
        mae   = np.mean(np.abs(data["gt"] - data["pred"]))
        color = profile_colors[profile]

        ax.plot(t, data["gt"],   color="#16a34a", linewidth=2,   label="Ground Truth")
        ax.plot(t, data["pred"], color=color,     linewidth=1.5, linestyle="--", label="Model Rollout")

        t_f = t[:len(data["F_ext"])]
        ax2 = ax.twinx()
        ax2.fill_between(t_f, data["F_ext"], alpha=0.12, color=color)
        ax2.plot(t_f, data["F_ext"], color=color, linewidth=0.8, alpha=0.5)
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
# 12.  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    TRUE_PARAMS    = PhysicsParams(b=0.8, mu_k=0.25, mu_s=0.30, g=9.81, dt=0.02)
    ROLLOUT_STEPS  = 10
    WEIGHT_MODE    = "linear"     # "uniform" | "linear" | "exp_decay"
    CURRICULUM     = 50           # epochs to ramp from K=1 to K=10

    print("Generating dataset...")
    pos_seqs, fext_seqs, masses = generate_dataset(
        n_trajectories=600,
        n_steps=100,
        rollout_steps=ROLLOUT_STEPS,
        params=TRUE_PARAMS,
    )

    n_total = pos_seqs.shape[0]
    n_train = int(0.85 * n_total)
    perm    = torch.randperm(n_total)

    pos_train  = pos_seqs[perm[:n_train]]
    fext_train = fext_seqs[perm[:n_train]]
    mass_train = masses[perm[:n_train]]

    pos_val    = pos_seqs[perm[n_train:]]
    fext_val   = fext_seqs[perm[n_train:]]
    mass_val   = masses[perm[n_train:]]

    print(f"Dataset: {n_train} train, {n_total - n_train} val sequences  (K={ROLLOUT_STEPS})\n")

    print(f"Training with rollout_steps={ROLLOUT_STEPS}, weight_mode={WEIGHT_MODE}, "
          f"curriculum={CURRICULUM} epochs...")
    model = PhysicsMotionModel(dt_init=0.05, hidden_dim=32, n_hidden=2)
    history = train(
        model,
        pos_train, fext_train, mass_train,
        pos_val,   fext_val,   mass_val,
        rollout_steps=ROLLOUT_STEPS,
        weight_mode=WEIGHT_MODE,
        curriculum_epochs=CURRICULUM,
        n_epochs=300,
        lr=1e-3,
    )

    print(f"\nLearned dt: {model.dt.item():.4f}  (true: {TRUE_PARAMS.dt:.4f})")

    rng = np.random.default_rng(99)
    rollout_data = {}
    print("\nRollout MAE by profile:")
    for profile in PROFILES:
        F_ext = make_force_profile(profile, 150, TRUE_PARAMS.dt, rng)
        gt, pred = rollout(model, x0=6.0, v0=-4.0, mass=2.0,
                           F_ext_series=F_ext, params=TRUE_PARAMS)
        mae = np.mean(np.abs(gt - pred))
        print(f"  {profile:<15} MAE = {mae:.5f} m")
        rollout_data[profile] = {"model": model, "gt": gt, "pred": pred, "F_ext": F_ext}

    plot_results(history, rollout_data, TRUE_PARAMS)
