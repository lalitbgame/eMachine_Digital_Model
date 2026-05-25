"""
=============================================================================
  EV E-Machine Electromagnetic Force Prediction
  Pipeline: FEA Data → POD (SVD) → ANN Surrogate Model
  
  Machine:  48-slot stator (industrial IPMSM / PMSM typical)
  Forces:   Radial (Fr) and Tangential (Ft) per tooth
  Inputs:   Speed [RPM], Torque [Nm], Electrical angle θ [deg]
  Output:   Fr(θ), Ft(θ) on each of 48 teeth
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit these to match your data)
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "n_teeth":          48,          # stator teeth
    "n_force_comp":     2,           # [Fr, Ft]
    "speed_rpm":        1000,        # fixed speed
    "torque_values":    [10, 20, 30, 40, 50, 60, 70, 80, 100, 120],  # [Nm]
    "n_angles":         360,         # electrical angle steps per cycle
    "pod_energy_thresh": 0.9999,     # retain modes until this energy fraction
    "pod_max_modes":    20,          # hard cap on number of modes
    "ann_hidden":       [128, 256, 128, 64],  # hidden layer sizes
    "ann_epochs":       500,
    "ann_lr":           1e-3,
    "ann_batch_size":   256,
    "ann_dropout":      0.15,
    "train_split":      0.75,
    "val_split":        0.15,        # remainder = test
    "data_path":        None,        # set to Path("your_data.csv") or None for demo
    "output_dir":       Path("pod_ann_results"),
    "random_seed":      42,
}
torch.manual_seed(CFG["random_seed"])
np.random.seed(CFG["random_seed"])
CFG["output_dir"].mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(cfg: dict) -> dict:
    """
    Load FEA / measured data.
    
    Expected CSV columns (one row = one snapshot):
        speed_rpm, torque_nm, angle_deg,
        Fr_t1, Fr_t2, ..., Fr_t48,   (radial force per tooth, N/m)
        Ft_t1, Ft_t2, ..., Ft_t48    (tangential force per tooth, N/m)
    
    Returns a dict with:
        X      : snapshot matrix  [2*n_teeth × n_snapshots]
        inputs : operating conds  [n_snapshots × 3]  (rpm, torque, angle)
    """
    if cfg["data_path"] is not None and Path(cfg["data_path"]).exists():
        df = pd.read_csv(cfg["data_path"])
        fr_cols = [f"Fr_t{i+1}" for i in range(cfg["n_teeth"])]
        ft_cols = [f"Ft_t{i+1}" for i in range(cfg["n_teeth"])]
        Fr = df[fr_cols].values.T           # [48 × N_snap]
        Ft = df[ft_cols].values.T           # [48 × N_snap]
        inputs = df[["speed_rpm", "torque_nm", "angle_deg"]].values
    else:
        print("  [INFO] No data path set — generating synthetic demo data.")
        Fr, Ft, inputs = _generate_synthetic_data(cfg)

    X = np.vstack([Fr, Ft])  # [96 × N_snap]
    print(f"  Snapshot matrix X shape : {X.shape}")
    print(f"  Input conditions shape  : {inputs.shape}")
    return {"X": X, "inputs": inputs, "Fr": Fr, "Ft": Ft}


def _generate_synthetic_data(cfg):
    """
    Synthetic eMag force data with:
    - Dominant fundamental + harmonics (6th, 12th, 24th) typical for 48-slot
    - Torque-dependent amplitude scaling
    - Spatial variation across 48 teeth (pole-pitch periodicity)
    """
    n_t   = cfg["n_teeth"]
    n_ang = cfg["n_angles"]
    torques = cfg["torque_values"]
    angles  = np.linspace(0, 360, n_ang, endpoint=False)
    tooth_idx = np.arange(n_t)

    # Spatial tooth phase (mechanical)
    tooth_phase = 2 * np.pi * tooth_idx / n_t  # 4 pole pairs → period 12

    Fr_list, Ft_list, inp_list = [], [], []

    for torq in torques:
        for ang_deg in angles:
            ang_rad = np.deg2rad(ang_deg)
            # Radial: strong DC + harmonics
            Fr_tooth = (
                  (800 + 5.0 * torq)                                      # DC offset
                + (200 + 1.5 * torq) * np.cos( 1 * tooth_phase + ang_rad)
                + (80  + 0.6 * torq) * np.cos( 6 * tooth_phase + 6*ang_rad)
                + (40  + 0.3 * torq) * np.cos(12 * tooth_phase +12*ang_rad)
                + (20  + 0.1 * torq) * np.cos(24 * tooth_phase +24*ang_rad)
                + 5 * np.random.randn(n_t)
            )
            # Tangential: smaller, mostly AC
            Ft_tooth = (
                  (0.8 * torq) * np.sin( 1 * tooth_phase + ang_rad)
                + (0.3 * torq) * np.sin( 6 * tooth_phase + 6*ang_rad)
                + (0.1 * torq) * np.sin(12 * tooth_phase +12*ang_rad)
                + 2 * np.random.randn(n_t)
            )
            Fr_list.append(Fr_tooth)
            Ft_list.append(Ft_tooth)
            inp_list.append([cfg["speed_rpm"], torq, ang_deg])

    Fr = np.array(Fr_list).T      # [48 × N_snap]
    Ft = np.array(Ft_list).T      # [48 × N_snap]
    inputs = np.array(inp_list)   # [N_snap × 3]
    return Fr, Ft, inputs


# ─────────────────────────────────────────────────────────────────────────────
# 2. POD / SVD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
class PODReducer:
    """
    Proper Orthogonal Decomposition via truncated SVD.
    
    Workflow:
      fit()      → compute modes from training snapshots
      transform()→ project any snapshot onto modal coordinates
      reconstruct()→ map modal coords back to physical space
    """
    def __init__(self, energy_thresh=0.9999, max_modes=20):
        self.energy_thresh = energy_thresh
        self.max_modes     = max_modes
        self.modes_        = None   # Φ : [2n_teeth × r]
        self.singular_     = None   # σ values
        self.n_modes_      = None
        self.mean_         = None   # column mean for centering

    def fit(self, X: np.ndarray):
        """X : [2*n_teeth × N_snapshots]"""
        self.mean_ = X.mean(axis=1, keepdims=True)
        Xc = X - self.mean_                      # centre
        U, s, Vt = svd(Xc, full_matrices=False)  # economy SVD
        self.singular_ = s

        # Choose r by cumulative energy
        energy = np.cumsum(s**2) / np.sum(s**2)
        r = int(np.searchsorted(energy, self.energy_thresh)) + 1
        r = min(r, self.max_modes)
        self.n_modes_ = r
        self.modes_   = U[:, :r]   # Φ

        captured = energy[r-1] * 100
        print(f"\n  POD: retaining {r} modes  ({captured:.4f}% energy)")
        print(f"  Singular values (top 10): {s[:10].round(2)}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X → modal amplitudes  [N_snap × r]"""
        Xc = X - self.mean_
        return (self.modes_.T @ Xc).T

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        """modal amplitudes [N_snap × r] → physical [2*n_teeth × N_snap]"""
        return (self.modes_ @ a.T) + self.mean_

    def relative_error(self, X_orig, X_recon):
        err = np.linalg.norm(X_orig - X_recon, "fro")
        ref = np.linalg.norm(X_orig, "fro")
        return err / ref * 100


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANN SURROGATE MODEL
# ─────────────────────────────────────────────────────────────────────────────
class EmagForceANN(nn.Module):
    """
    Feedforward ANN:
      Input  : [RPM, Torque, θ_elec]  → normalised
      Output : [a₁, a₂, ..., aᵣ]      → POD modal amplitudes
    
    Uses SELU activations + AlphaDropout (self-normalising).
    A skip connection from input to output improves convergence for
    near-linear torque scaling.
    """
    def __init__(self, n_modes: int, hidden: list, dropout: float = 0.15):
        super().__init__()
        layers = []
        in_dim = 3
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.SELU(), nn.AlphaDropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_modes))
        self.net    = nn.Sequential(*layers)
        self.skip   = nn.Linear(3, n_modes, bias=False)   # linear path

    def forward(self, x):
        return self.net(x) + self.skip(x)


class AnnTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  ANN device: {self.device}")

    def build_dataset(self, inputs: np.ndarray, amplitudes: np.ndarray):
        """
        inputs     : [N × 3]  (rpm, torque, angle)
        amplitudes : [N × r]  POD modal coords
        """
        self.in_scaler  = StandardScaler()
        self.out_scaler = StandardScaler()

        X_sc = self.in_scaler.fit_transform(inputs).astype(np.float32)
        y_sc = self.out_scaler.fit_transform(amplitudes).astype(np.float32)

        dataset = TensorDataset(torch.tensor(X_sc), torch.tensor(y_sc))
        n  = len(dataset)
        n_tr = int(n * self.cfg["train_split"])
        n_va = int(n * self.cfg["val_split"])
        n_te = n - n_tr - n_va

        g = torch.Generator().manual_seed(self.cfg["random_seed"])
        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset, [n_tr, n_va, n_te], generator=g
        )
        print(f"  Dataset splits — Train:{n_tr}  Val:{n_va}  Test:{n_te}")

    def train(self, n_modes: int):
        cfg = self.cfg
        model = EmagForceANN(n_modes, cfg["ann_hidden"], cfg["ann_dropout"]).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg["ann_lr"], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["ann_epochs"])
        criterion = nn.MSELoss()

        tr_loader = DataLoader(self.train_ds, batch_size=cfg["ann_batch_size"], shuffle=True)
        va_loader = DataLoader(self.val_ds,   batch_size=512, shuffle=False)

        best_val, best_state, history = np.inf, None, {"train": [], "val": []}

        for epoch in range(1, cfg["ann_epochs"] + 1):
            model.train()
            tr_loss = sum(
                criterion(model(xb.to(self.device)), yb.to(self.device)).item()
                for xb, yb in tr_loader
            ) / len(tr_loader)

            model.eval()
            with torch.no_grad():
                va_loss = sum(
                    criterion(model(xb.to(self.device)), yb.to(self.device)).item()
                    for xb, yb in va_loader
                ) / len(va_loader)

            history["train"].append(tr_loss)
            history["val"].append(va_loss)
            scheduler.step()

            if va_loss < best_val:
                best_val  = va_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 50 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{cfg['ann_epochs']} | "
                      f"Train: {tr_loss:.5f} | Val: {va_loss:.5f}")

        model.load_state_dict(best_state)
        self.model   = model
        self.history = history
        print(f"\n  Best validation loss : {best_val:.6f}")
        return model

    def predict(self, inputs_raw: np.ndarray) -> np.ndarray:
        """inputs_raw [N×3] → predicted modal amps [N×r] (un-scaled)"""
        self.model.eval()
        X_sc = self.in_scaler.transform(inputs_raw).astype(np.float32)
        with torch.no_grad():
            y_sc = self.model(torch.tensor(X_sc).to(self.device)).cpu().numpy()
        return self.out_scaler.inverse_transform(y_sc)

    def evaluate(self, pod: PODReducer, X_phys: np.ndarray, inputs: np.ndarray):
        """Full pipeline evaluation on test set."""
        te_loader = DataLoader(self.test_ds, batch_size=512)
        all_xb, all_yb, all_yh = [], [], []

        self.model.eval()
        with torch.no_grad():
            for xb, yb in te_loader:
                yh = self.model(xb.to(self.device)).cpu()
                all_xb.append(xb);  all_yb.append(yb);  all_yh.append(yh)

        yb_np  = torch.cat(all_yb).numpy()
        yh_np  = torch.cat(all_yh).numpy()

        # Un-scale
        amp_true = self.out_scaler.inverse_transform(yb_np)
        amp_pred = self.out_scaler.inverse_transform(yh_np)

        # Reconstruct physical forces
        X_true_recon = pod.reconstruct(amp_true)
        X_pred_recon = pod.reconstruct(amp_pred)

        r2  = r2_score(X_true_recon.ravel(), X_pred_recon.ravel())
        rms = np.sqrt(mean_squared_error(X_true_recon.ravel(), X_pred_recon.ravel()))
        rel = np.linalg.norm(X_true_recon - X_pred_recon, "fro") / \
              np.linalg.norm(X_true_recon, "fro") * 100

        print(f"\n  ── Test-set evaluation (physical space) ──")
        print(f"     R²    : {r2:.6f}")
        print(f"     RMSE  : {rms:.4f} N/m")
        print(f"     Rel.ε : {rel:.4f} %")
        return {"r2": r2, "rmse": rms, "rel_err": rel,
                "X_true": X_true_recon, "X_pred": X_pred_recon}


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_pod_analysis(pod: PODReducer, cfg: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("POD Analysis — E-machine eMag Forces", fontsize=13, fontweight="bold")

    # Singular value spectrum
    ax = axes[0]
    s  = pod.singular_
    ax.semilogy(np.arange(1, len(s)+1), s, "o-", ms=4, color="#1D9E75")
    ax.axvline(pod.n_modes_, color="#D85A30", ls="--", lw=1.5,
               label=f"r = {pod.n_modes_} retained")
    ax.set_xlabel("Mode index"); ax.set_ylabel("Singular value (log)")
    ax.set_title("Singular value spectrum"); ax.legend(); ax.grid(alpha=0.3)

    # Cumulative energy
    ax = axes[1]
    energy = np.cumsum(s**2) / np.sum(s**2) * 100
    ax.plot(np.arange(1, len(s)+1), energy, "s-", ms=4, color="#534AB7")
    ax.axvline(pod.n_modes_, color="#D85A30", ls="--", lw=1.5)
    ax.axhline(pod.energy_thresh*100, color="#888", ls=":", lw=1)
    ax.set_xlabel("Mode index"); ax.set_ylabel("Cumulative energy (%)")
    ax.set_title("Energy capture"); ax.grid(alpha=0.3)

    # First 4 POD spatial modes (radial component, 48 teeth)
    ax = axes[2]
    n = cfg["n_teeth"]
    colors = ["#1D9E75","#534AB7","#D85A30","#185FA5"]
    for i in range(min(4, pod.n_modes_)):
        mode_fr = pod.modes_[:n, i]  # radial part
        ax.plot(np.arange(n), mode_fr / np.max(np.abs(mode_fr)),
                label=f"Mode {i+1}", color=colors[i], lw=1.5)
    ax.set_xlabel("Tooth index"); ax.set_ylabel("Normalised amplitude")
    ax.set_title("POD spatial modes (Fr)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "pod_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_dir / 'pod_analysis.png'}")


def plot_training_history(history: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = np.arange(1, len(history["train"])+1)
    ax.semilogy(epochs, history["train"], label="Train loss", color="#1D9E75")
    ax.semilogy(epochs, history["val"],   label="Val loss",   color="#D85A30")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (log)")
    ax.set_title("ANN Training History"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_dir / 'training_history.png'}")


def plot_force_prediction(eval_results: dict, cfg: dict, out_dir: Path,
                          n_teeth: int = 48, sample_snap: int = 0):
    X_true = eval_results["X_true"]
    X_pred = eval_results["X_pred"]
    n = cfg["n_teeth"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"eMag Force Prediction vs FEA — R² = {eval_results['r2']:.5f}",
                 fontsize=13, fontweight="bold")

    tooth_idx = np.arange(n)
    snap = sample_snap  # which snapshot to plot

    # Fr per tooth (single snapshot)
    ax = axes[0, 0]
    ax.plot(tooth_idx, X_true[:n, snap], "o-", ms=4, label="FEA (truth)", color="#185FA5")
    ax.plot(tooth_idx, X_pred[:n, snap], "s--", ms=4, label="ANN-POD", color="#D85A30")
    ax.set_xlabel("Tooth index"); ax.set_ylabel("Fr [N/m]")
    ax.set_title(f"Radial force — snapshot {snap}"); ax.legend(); ax.grid(alpha=0.3)

    # Ft per tooth (single snapshot)
    ax = axes[0, 1]
    ax.plot(tooth_idx, X_true[n:, snap], "o-", ms=4, label="FEA (truth)", color="#1D9E75")
    ax.plot(tooth_idx, X_pred[n:, snap], "s--", ms=4, label="ANN-POD", color="#D85A30")
    ax.set_xlabel("Tooth index"); ax.set_ylabel("Ft [N/m]")
    ax.set_title(f"Tangential force — snapshot {snap}"); ax.legend(); ax.grid(alpha=0.3)

    # Parity plot — Fr
    ax = axes[1, 0]
    ax.scatter(X_true[:n].ravel(), X_pred[:n].ravel(),
               s=1, alpha=0.3, color="#185FA5")
    lim = [X_true[:n].min(), X_true[:n].max()]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("FEA Fr [N/m]"); ax.set_ylabel("Predicted Fr [N/m]")
    ax.set_title("Parity plot — radial force"); ax.grid(alpha=0.3)

    # Parity plot — Ft
    ax = axes[1, 1]
    ax.scatter(X_true[n:].ravel(), X_pred[n:].ravel(),
               s=1, alpha=0.3, color="#1D9E75")
    lim = [X_true[n:].min(), X_true[n:].max()]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("FEA Ft [N/m]"); ax.set_ylabel("Predicted Ft [N/m]")
    ax.set_title("Parity plot — tangential force"); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "force_prediction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_dir / 'force_prediction.png'}")


def plot_polar_forces(eval_results: dict, cfg: dict, out_dir: Path, snap: int = 0):
    """Polar plot of Fr / Ft on all 48 teeth."""
    X_true = eval_results["X_true"]
    X_pred = eval_results["X_pred"]
    n = cfg["n_teeth"]
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    theta = np.append(theta, theta[0])  # close loop

    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(12, 5))
    fig.suptitle(f"Polar tooth-force distribution — snapshot {snap}", fontsize=12)

    for ax, label, sl in zip(axes, ["Fr (radial)", "Ft (tangential)"],
                              [slice(None, n), slice(n, None)]):
        tr = np.append(X_true[sl, snap], X_true[sl, snap][0])
        pr = np.append(X_pred[sl, snap], X_pred[sl, snap][0])
        ax.plot(theta, tr, lw=1.8, label="FEA", color="#185FA5")
        ax.plot(theta, pr, lw=1.4, ls="--", label="ANN-POD", color="#D85A30")
        ax.set_title(label, pad=14); ax.legend(loc="upper right", fontsize=8)
        ax.set_theta_zero_location("N")

    plt.tight_layout()
    fig.savefig(out_dir / "polar_forces.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_dir / 'polar_forces.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. INFERENCE UTILITY
# ─────────────────────────────────────────────────────────────────────────────
class EmagForcePredictor:
    """
    After training, wrap pod + ann_trainer for single-query inference.
    
    Usage:
        pred = EmagForcePredictor(pod, trainer, cfg)
        Fr, Ft = pred.predict(torque_nm=60, angle_deg=45)
    """
    def __init__(self, pod: PODReducer, trainer: AnnTrainer, cfg: dict):
        self.pod     = pod
        self.trainer = trainer
        self.cfg     = cfg

    def predict(self, torque_nm: float, angle_deg: float,
                speed_rpm: float = None) -> tuple:
        """
        Returns Fr [48], Ft [48] in N/m.
        """
        rpm = speed_rpm or self.cfg["speed_rpm"]
        inp = np.array([[rpm, torque_nm, angle_deg]], dtype=np.float32)
        amp = self.trainer.predict(inp)           # [1 × r]
        xhat = self.pod.reconstruct(amp)          # [96 × 1]
        Fr   = xhat[:self.cfg["n_teeth"], 0]
        Ft   = xhat[self.cfg["n_teeth"]:, 0]
        return Fr, Ft

    def force_over_cycle(self, torque_nm: float,
                         n_steps: int = 360) -> dict:
        """
        Predict Fr, Ft for all 48 teeth over one full electrical cycle.
        Returns dict with keys 'Fr' [48 × n_steps], 'Ft' [48 × n_steps].
        """
        angles  = np.linspace(0, 360, n_steps, endpoint=False)
        rpm_col = np.full(n_steps, self.cfg["speed_rpm"])
        tq_col  = np.full(n_steps, torque_nm)
        inp     = np.column_stack([rpm_col, tq_col, angles])
        amp     = self.trainer.predict(inp)       # [n_steps × r]
        Xhat    = self.pod.reconstruct(amp)       # [96 × n_steps]
        n       = self.cfg["n_teeth"]
        return {"Fr": Xhat[:n], "Ft": Xhat[n:], "angles": angles}


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  EV E-Machine eMag Force: POD + ANN Pipeline")
    print("=" * 65)

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n[1/5] Loading data...")
    data = load_data(CFG)
    X, inputs = data["X"], data["inputs"]

    # ── Step 2: POD ────────────────────────────────────────────
    print("\n[2/5] Running POD (SVD)...")
    pod = PODReducer(energy_thresh=CFG["pod_energy_thresh"],
                     max_modes=CFG["pod_max_modes"])
    pod.fit(X)

    amplitudes = pod.transform(X)   # [N_snap × r]
    X_pod_recon = pod.reconstruct(amplitudes)
    pod_err = pod.relative_error(X, X_pod_recon)
    print(f"  POD reconstruction error (full dataset): {pod_err:.4f} %")

    plot_pod_analysis(pod, CFG, CFG["output_dir"])

    # ── Step 3: ANN training ───────────────────────────────────
    print("\n[3/5] Training ANN surrogate...")
    trainer = AnnTrainer(CFG)
    trainer.build_dataset(inputs, amplitudes)
    trainer.train(pod.n_modes_)
    plot_training_history(trainer.history, CFG["output_dir"])

    # ── Step 4: Evaluation ─────────────────────────────────────
    print("\n[4/5] Evaluating on held-out test set...")
    eval_res = trainer.evaluate(pod, X, inputs)
    plot_force_prediction(eval_res, CFG, CFG["output_dir"])
    plot_polar_forces(eval_res, CFG, CFG["output_dir"])

    # ── Step 5: Save & wrap predictor ─────────────────────────
    print("\n[5/5] Saving model & demo prediction...")
    torch.save({
        "model_state": trainer.model.state_dict(),
        "in_scaler":   trainer.in_scaler,
        "out_scaler":  trainer.out_scaler,
        "pod_modes":   pod.modes_,
        "pod_mean":    pod.mean_,
        "pod_singular":pod.singular_,
        "n_modes":     pod.n_modes_,
        "cfg":         CFG,
    }, CFG["output_dir"] / "pod_ann_model.pt")
    print(f"  Model saved → {CFG['output_dir'] / 'pod_ann_model.pt'}")

    # Demo: predict forces at one operating point
    predictor = EmagForcePredictor(pod, trainer, CFG)
    Fr_demo, Ft_demo = predictor.predict(torque_nm=60, angle_deg=90.0)
    print(f"\n  Demo prediction  (T=60 Nm, θ=90°):")
    print(f"    Fr max/min: {Fr_demo.max():.1f} / {Fr_demo.min():.1f} N/m")
    print(f"    Ft max/min: {Ft_demo.max():.1f} / {Ft_demo.min():.1f} N/m")

    # Full cycle prediction
    cycle = predictor.force_over_cycle(torque_nm=60)
    print(f"\n  Force cycle shape: Fr {cycle['Fr'].shape}, Ft {cycle['Ft'].shape}")
    print(f"  (48 teeth × {len(cycle['angles'])} angle steps)")

    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print(f"  Results in: {CFG['output_dir'].resolve()}")
    print("=" * 65)

    return predictor


if __name__ == "__main__":
    predictor = main()


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO LOAD YOUR OWN DATA
# ─────────────────────────────────────────────────────────────────────────────
"""
Option A — CSV format (one row per snapshot):
    speed_rpm, torque_nm, angle_deg, Fr_t1, ..., Fr_t48, Ft_t1, ..., Ft_t48

    CFG["data_path"] = Path("my_fea_data.csv")

Option B — NumPy arrays directly (edit load_data):
    Fr : np.ndarray [48 × N_snap]   radial forces, N/m
    Ft : np.ndarray [48 × N_snap]   tangential forces, N/m
    inputs : np.ndarray [N_snap × 3]  [rpm, torque, angle_deg]

Option C — Multiple torque files:
    frames = []
    for T in [10, 20, 30, 40, 50, 60, 70, 80, 100, 120]:
        df = pd.read_csv(f"torque_{T}Nm.csv")
        df["torque_nm"] = T;  df["speed_rpm"] = 1000
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv("combined_data.csv", index=False)
    CFG["data_path"] = Path("combined_data.csv")

HYPERPARAMETER TUNING GUIDE:
─────────────────────────────
  pod_energy_thresh   0.999 → aggressive compression, 0.9999 → near-lossless
  pod_max_modes       Start 20, inspect singular value plot — choose elbow
  ann_hidden          Deeper for strong nonlinearity; [64,128,64] for sparse data
  ann_epochs          Monitor val loss; early stop if plateau
  ann_lr              1e-3 default; try 3e-4 if unstable
  ann_dropout         0.1–0.2; reduce to 0.05 for small datasets
"""
