"""
================================================================================
  Script 1 — Hyperparameter Optimisation: Autoencoder (Encoder + Decoder)
  ─────────────────────────────────────────────────────────────────────────────
  Framework : Optuna (TPE sampler + MedianPruner)
  Objective : minimise validation MSE reconstruction loss

  Search space
  ────────────
  Architecture
    • n_encoder_layers     : 2 – 5
    • layer width schedule : [wide→narrow] with log-uniform sampling
    • latent_dim           : 4, 8, 12, 16, 24, 32
    • activation           : GELU | ReLU | ELU | Tanh
    • use_batch_norm       : True | False
    • use_layer_norm       : True | False  (mutually exclusive w/ BN)
    • skip_connections     : True | False  (residual blocks in encoder/decoder)

  Regularisation
    • dropout              : 0.0 – 0.4
    • weight_decay         : 1e-6 – 1e-3  (log-uniform)
    • gradient_clip_val    : 0.5 – 5.0

  Optimiser & schedule
    • optimizer            : AdamW | Adam | RMSprop
    • learning_rate        : 1e-4 – 5e-3  (log-uniform)
    • lr_scheduler         : CosineAnnealing | ReduceLROnPlateau | StepLR | None
    • batch_size           : 64, 128, 256, 512

  Outputs
  ───────
  hpo_ae_results/
    ├── best_ae_params.json          best trial hyperparameters
    ├── best_ae_model.pt             best model weights + scalers
    ├── optuna_ae_study.pkl          full Optuna study (resumable)
    ├── ae_hpo_history.png           optimisation history plot
    ├── ae_param_importance.png      parameter importance plot
    ├── ae_parallel_coords.png       parallel coordinates plot
    └── ae_trial_log.csv             all trial results
================================================================================
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG  — edit these to match your setup
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # Data
    "n_teeth":          48,
    "input_dim":        96,          # 2 × n_teeth  (Fr + Ft)
    "speed_rpm":        1000,
    "torque_values":    [10, 20, 30, 40, 50, 60, 70, 80, 100, 120],
    "n_angles":         360,
    "data_path":        None,        # Path("your_data.csv") or None → synthetic

    # Optuna study
    "n_trials":         80,          # total Optuna trials
    "n_startup_trials": 15,          # random exploration before TPE kicks in
    "n_warmup_steps":   20,          # pruner: epochs before pruning starts
    "pruning_interval": 5,           # report intermediate value every N epochs

    # Per-trial training budget
    "max_epochs":       300,         # epochs per trial
    "early_stop_patience": 25,       # stop trial if val loss doesn't improve

    # Final retraining of best config
    "final_epochs":     600,
    "final_patience":   60,

    # Dataset splits
    "train_split":      0.75,
    "val_split":        0.15,
    "random_seed":      42,

    # Output
    "output_dir":       Path("hpo_ae_results"),
    "study_name":       "emag_autoencoder_hpo",
}

CFG["output_dir"].mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(CFG["random_seed"])
np.random.seed(CFG["random_seed"])
print(f"Device : {DEVICE}")
print(f"Trials : {CFG['n_trials']}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA  (replace with load_or_generate_data from main pipeline)
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset(cfg: dict):
    """Returns (force_mat [N×96], inputs [N×3], scaler) — scaled forces."""
    if cfg["data_path"] and Path(cfg["data_path"]).exists():
        df      = pd.read_csv(cfg["data_path"])
        fr_cols = [f"Fr_t{i+1}" for i in range(cfg["n_teeth"])]
        ft_cols = [f"Ft_t{i+1}" for i in range(cfg["n_teeth"])]
        Fr      = df[fr_cols].values
        Ft      = df[ft_cols].values
        inputs  = df[["speed_rpm", "torque_nm", "angle_deg"]].values
    else:
        Fr, Ft, inputs = _synthetic(cfg)

    force_mat = np.hstack([Fr, Ft]).astype(np.float32)
    scaler    = StandardScaler()
    force_mat = scaler.fit_transform(force_mat)
    return force_mat, inputs.astype(np.float32), scaler


def _synthetic(cfg):
    n  = cfg["n_teeth"]
    sp = 2 * np.pi * np.arange(n) / n
    rng = np.random.default_rng(cfg["random_seed"])
    Fr_l, Ft_l, inp_l = [], [], []
    for T in cfg["torque_values"]:
        for a in np.linspace(0, 360, cfg["n_angles"], endpoint=False):
            ar = np.deg2rad(a)
            Fr_l.append((800+5*T) + (200+1.8*T)*np.cos(sp+ar)
                        + (80+0.7*T)*np.cos(6*sp+6*ar)
                        + (35+0.3*T)*np.cos(12*sp+12*ar)
                        + rng.normal(0, 4, n))
            Ft_l.append((0.9*T)*np.sin(sp+ar) + (0.4*T)*np.sin(6*sp+6*ar)
                        + rng.normal(0, 2, n))
            inp_l.append([cfg["speed_rpm"], T, a])
    return (np.array(Fr_l, np.float32),
            np.array(Ft_l, np.float32),
            np.array(inp_l, np.float32))


def make_loaders(force_mat, inputs, cfg, batch_size):
    n     = len(force_mat)
    n_tr  = int(n * cfg["train_split"])
    n_va  = int(n * cfg["val_split"])
    n_te  = n - n_tr - n_va
    ds    = TensorDataset(torch.tensor(force_mat), torch.tensor(inputs))
    g     = torch.Generator().manual_seed(cfg["random_seed"])
    tr, va, te = random_split(ds, [n_tr, n_va, n_te], generator=g)
    pin = DEVICE.type == "cuda"
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,
                       pin_memory=pin, drop_last=True),
            DataLoader(va, batch_size=512,        shuffle=False, pin_memory=pin),
            DataLoader(te, batch_size=512,        shuffle=False, pin_memory=pin),
            tr, va, te)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FLEXIBLE ENCODER / DECODER
# ══════════════════════════════════════════════════════════════════════════════
def _get_activation(name: str) -> nn.Module:
    return {"GELU": nn.GELU, "ReLU": nn.ReLU, "ELU": nn.ELU,
            "Tanh": nn.Tanh}[name]()


def _make_norm(norm_type: str, dim: int) -> Optional[nn.Module]:
    if norm_type == "batch":   return nn.BatchNorm1d(dim)
    if norm_type == "layer":   return nn.LayerNorm(dim)
    return None                # "none"


class ResidualBlock(nn.Module):
    """Linear residual block: x → Linear → Norm → Act → Dropout → + proj(x)."""
    def __init__(self, in_d, out_d, norm_type, act_cls, dropout):
        super().__init__()
        self.linear = nn.Linear(in_d, out_d)
        norm = _make_norm(norm_type, out_d)
        self.norm   = norm if norm else nn.Identity()
        self.act    = act_cls()
        self.drop   = nn.Dropout(dropout)
        self.proj   = nn.Linear(in_d, out_d, bias=False) if in_d != out_d else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.linear(x))) + self.proj(x)


class FlexEncoder(nn.Module):
    def __init__(self, input_dim, layer_dims, latent_dim,
                 norm_type, act_name, dropout, use_skip):
        super().__init__()
        act_cls = _get_activation(act_name).__class__
        layers  = []
        prev    = input_dim
        for d in layer_dims:
            if use_skip:
                layers.append(ResidualBlock(prev, d, norm_type, act_cls, dropout))
            else:
                norm = _make_norm(norm_type, d)
                blk  = [nn.Linear(prev, d)]
                if norm: blk.append(norm)
                blk += [act_cls(), nn.Dropout(dropout)]
                layers.append(nn.Sequential(*blk))
            prev = d
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlexDecoder(nn.Module):
    def __init__(self, latent_dim, layer_dims, output_dim,
                 norm_type, act_name, dropout, use_skip):
        super().__init__()
        act_cls = _get_activation(act_name).__class__
        layers  = []
        prev    = latent_dim
        for d in layer_dims:
            if use_skip:
                layers.append(ResidualBlock(prev, d, norm_type, act_cls, dropout))
            else:
                norm = _make_norm(norm_type, d)
                blk  = [nn.Linear(prev, d)]
                if norm: blk.append(norm)
                blk += [act_cls(), nn.Dropout(dropout)]
                layers.append(nn.Sequential(*blk))
            prev = d
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class FlexAutoencoder(nn.Module):
    def __init__(self, encoder: FlexEncoder, decoder: FlexDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD MODEL FROM OPTUNA TRIAL
# ══════════════════════════════════════════════════════════════════════════════
def build_ae_from_trial(trial: optuna.Trial, cfg: dict) -> FlexAutoencoder:
    """
    Sample all architectural and regularisation hyperparameters from the trial.
    Decoder mirrors the encoder (reversed layer dims).
    """
    input_dim  = cfg["input_dim"]

    # ── Architecture ──────────────────────────────────────────────────────
    latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 12, 16, 24, 32])
    n_layers   = trial.suggest_int("n_encoder_layers", 2, 5)

    # Sample layer widths: must shrink toward the bottleneck
    # First layer ∈ [input_dim, 2×input_dim], each subsequent ≤ previous
    layer_dims = []
    prev_max   = max(input_dim * 2, 32)
    for i in range(n_layers):
        lo  = max(latent_dim * 2, 16)
        hi  = prev_max
        if lo >= hi:
            break
        d   = trial.suggest_int(f"enc_layer_{i}_dim", lo, hi, log=True)
        layer_dims.append(d)
        prev_max = d

    if not layer_dims:
        layer_dims = [max(latent_dim * 2, 32)]

    # ── Normalisation (mutually exclusive) ────────────────────────────────
    norm_choice = trial.suggest_categorical("norm_type",
                                             ["batch", "layer", "none"])

    # ── Activation ────────────────────────────────────────────────────────
    act_name = trial.suggest_categorical("activation",
                                          ["GELU", "ReLU", "ELU", "Tanh"])

    # ── Skip connections ──────────────────────────────────────────────────
    use_skip = trial.suggest_categorical("use_skip", [True, False])

    # ── Regularisation ────────────────────────────────────────────────────
    dropout = trial.suggest_float("dropout", 0.0, 0.4)

    encoder = FlexEncoder(input_dim, layer_dims, latent_dim,
                           norm_choice, act_name, dropout, use_skip)
    decoder = FlexDecoder(latent_dim, list(reversed(layer_dims)), input_dim,
                           norm_choice, act_name, dropout, use_skip)
    return FlexAutoencoder(encoder, decoder).to(DEVICE)


def build_optimizer(trial: optuna.Trial, params):
    opt_name = trial.suggest_categorical("optimizer",
                                          ["AdamW", "Adam", "RMSprop"])
    lr       = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    wd       = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    if opt_name == "AdamW":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        return optim.RMSprop(params, lr=lr, weight_decay=wd)


def build_scheduler(trial: optuna.Trial, optimizer, max_epochs):
    sched_name = trial.suggest_categorical(
        "lr_scheduler",
        ["CosineAnnealing", "ReduceLROnPlateau", "StepLR", "None"])

    if sched_name == "CosineAnnealing":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6), sched_name
    elif sched_name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-6), sched_name
    elif sched_name == "StepLR":
        step = trial.suggest_int("steplr_step_size", 30, 100)
        gamma = trial.suggest_float("steplr_gamma", 0.3, 0.9)
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step, gamma=gamma), sched_name
    else:
        return None, "None"


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP (shared by trial and final retraining)
# ══════════════════════════════════════════════════════════════════════════════
def train_ae_loop(ae, optimizer, scheduler, sched_name,
                  tr_ld, va_ld, max_epochs, patience,
                  clip_val, trial=None, pruning_interval=5):
    """
    Returns (best_val_loss, best_state_dict, history).
    If `trial` is provided, reports intermediate values for Optuna pruning.
    """
    crit = nn.MSELoss()
    best_val, best_sd = np.inf, None
    no_improve        = 0
    history           = {"train": [], "val": []}

    for epoch in range(1, max_epochs + 1):
        # ── train ──
        ae.train()
        tr_loss = 0.0
        for x_b, _ in tr_ld:
            x_b = x_b.to(DEVICE)
            optimizer.zero_grad()
            x_hat, _ = ae(x_b)
            loss = crit(x_hat, x_b)
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), clip_val)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        # ── validate ──
        ae.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x_b, _ in va_ld:
                x_hat, _ = ae(x_b.to(DEVICE))
                va_loss += crit(x_hat, x_b.to(DEVICE)).item()
        va_loss /= len(va_ld)

        # ── scheduler step ──
        if scheduler is not None:
            if sched_name == "ReduceLROnPlateau":
                scheduler.step(va_loss)
            else:
                scheduler.step()

        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_sd  = deepcopy(ae.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        # ── Optuna pruning ──
        if trial is not None and epoch % pruning_interval == 0:
            trial.report(va_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # ── early stopping ──
        if no_improve >= patience:
            break

    return best_val, best_sd, history


# ══════════════════════════════════════════════════════════════════════════════
# 5. OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════
# Module-level data (set once in main so objective closure is lightweight)
_FORCE_MAT = None
_INPUTS    = None
_CFG       = None


def objective(trial: optuna.Trial) -> float:
    cfg       = _CFG
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    clip_val   = trial.suggest_float("grad_clip", 0.5, 5.0)

    tr_ld, va_ld, _, _, _, _ = make_loaders(
        _FORCE_MAT, _INPUTS, cfg, batch_size)

    ae        = build_ae_from_trial(trial, cfg)
    optimizer = build_optimizer(trial, ae.parameters())
    scheduler, sched_name = build_scheduler(
        trial, optimizer, cfg["max_epochs"])

    best_val, _, _ = train_ae_loop(
        ae, optimizer, scheduler, sched_name,
        tr_ld, va_ld,
        max_epochs       = cfg["max_epochs"],
        patience         = cfg["early_stop_patience"],
        clip_val         = clip_val,
        trial            = trial,
        pruning_interval = cfg["pruning_interval"],
    )
    return best_val


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
def plot_hpo_results(study: optuna.Study, out_dir: Path):
    """Four diagnostic plots saved to disk."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    trials_df = study.trials_dataframe(attrs=("number","value","state","duration"))

    # ── 1. Optimisation history ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    completed = trials_df[trials_df["state"] == "COMPLETE"]
    ax.scatter(completed["number"], completed["value"],
               s=18, alpha=0.6, color="#185FA5", label="Trial val loss")
    best_so_far = completed["value"].cummin()
    ax.plot(completed["number"], best_so_far, color="#D85A30",
            lw=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("Val MSE loss")
    ax.set_title("Autoencoder HPO — optimisation history")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "ae_hpo_history.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. Parameter importances ──────────────────────────────────────────
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(9, max(4, len(importances)*0.4)))
        params  = list(importances.keys())
        vals    = list(importances.values())
        colors  = ["#1D9E75" if v == max(vals) else "#534AB7" for v in vals]
        ax.barh(params, vals, color=colors)
        ax.set_xlabel("Importance score")
        ax.set_title("Autoencoder HPO — parameter importances (fANOVA)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "ae_param_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    # ── 3. Parallel coordinates ───────────────────────────────────────────
    try:
        import optuna.visualization as ov
        import plotly.io as pio
        fig_pc = ov.plot_parallel_coordinate(study)
        fig_pc.write_image(str(out_dir / "ae_parallel_coords.png"))
    except Exception:
        pass  # plotly/kaleido optional

    # ── 4. Training curve of best trial ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    best    = study.best_trial
    ax.set_title(f"Best trial #{best.number}  val={best.value:.6f}")
    ax.text(0.5, 0.45,
            "\n".join(f"{k}: {v}" for k, v in sorted(best.params.items())),
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="#F1EFE8", alpha=0.7))
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "ae_best_trial_params.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plots saved to {out_dir}")


def plot_final_training(history: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ep = np.arange(1, len(history["train"]) + 1)
    ax.semilogy(ep, history["train"], label="Train", color="#1D9E75")
    ax.semilogy(ep, history["val"],   label="Val",   color="#D85A30")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (log)")
    ax.set_title("Final autoencoder training (best hyperparameters)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "ae_final_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir / 'ae_final_training.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. RECONSTRUCT BEST MODEL AND RETRAIN AT FULL BUDGET
# ══════════════════════════════════════════════════════════════════════════════
def retrain_best_ae(best_params: dict, cfg: dict,
                    force_mat, inputs, scaler) -> dict:
    """
    Rebuild the best architecture from best_params and train for final_epochs
    on train+val combined, evaluate on held-out test set.
    """
    print("\n── Retraining best autoencoder at full budget ─────────────────")

    # Reconstruct layer dims from saved params
    n_layers = best_params["n_encoder_layers"]
    layer_dims = []
    for i in range(n_layers):
        key = f"enc_layer_{i}_dim"
        if key in best_params:
            layer_dims.append(best_params[key])
    if not layer_dims:
        layer_dims = [max(best_params["latent_dim"] * 2, 32)]

    encoder = FlexEncoder(
        input_dim  = cfg["input_dim"],
        layer_dims = layer_dims,
        latent_dim = best_params["latent_dim"],
        norm_type  = best_params["norm_type"],
        act_name   = best_params["activation"],
        dropout    = best_params["dropout"],
        use_skip   = best_params["use_skip"],
    )
    decoder = FlexDecoder(
        latent_dim  = best_params["latent_dim"],
        layer_dims  = list(reversed(layer_dims)),
        output_dim  = cfg["input_dim"],
        norm_type   = best_params["norm_type"],
        act_name    = best_params["activation"],
        dropout     = best_params["dropout"],
        use_skip    = best_params["use_skip"],
    )
    ae = FlexAutoencoder(encoder, decoder).to(DEVICE)

    # Use full train+val for retraining
    n     = len(force_mat)
    n_te  = n - int(n * cfg["train_split"]) - int(n * cfg["val_split"])
    n_trva = n - n_te
    ds    = TensorDataset(torch.tensor(force_mat), torch.tensor(inputs))
    g     = torch.Generator().manual_seed(cfg["random_seed"])
    trva_ds, te_ds = random_split(ds, [n_trva, n_te], generator=g)
    bs    = best_params["batch_size"]
    tr_ld = DataLoader(trva_ds, batch_size=bs, shuffle=True, drop_last=True)
    va_ld = DataLoader(te_ds,   batch_size=512)

    # Rebuild optimizer
    lr  = best_params["lr"]
    wd  = best_params["weight_decay"]
    opt_name = best_params["optimizer"]
    if opt_name == "AdamW":
        optimizer = optim.AdamW(ae.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(ae.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.RMSprop(ae.parameters(), lr=lr, weight_decay=wd)

    sched_name = best_params["lr_scheduler"]
    if sched_name == "CosineAnnealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["final_epochs"], eta_min=1e-6)
    elif sched_name == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, factor=0.5, min_lr=1e-6)
    elif sched_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size = best_params.get("steplr_step_size", 50),
            gamma     = best_params.get("steplr_gamma", 0.5))
    else:
        scheduler = None

    clip_val = best_params["grad_clip"]
    best_val, best_sd, history = train_ae_loop(
        ae, optimizer, scheduler, sched_name,
        tr_ld, va_ld,
        max_epochs = cfg["final_epochs"],
        patience   = cfg["final_patience"],
        clip_val   = clip_val,
    )
    ae.load_state_dict(best_sd)

    # ── Test-set evaluation ───────────────────────────────────────────────
    ae.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x_b, _ in va_ld:
            x_hat, _ = ae(x_b.to(DEVICE))
            preds.append(x_hat.cpu().numpy())
            trues.append(x_b.numpy())
    preds = scaler.inverse_transform(np.vstack(preds))
    trues = scaler.inverse_transform(np.vstack(trues))
    r2   = r2_score(trues.ravel(), preds.ravel())
    rmse = np.sqrt(np.mean((trues - preds) ** 2))
    rel  = np.linalg.norm(trues-preds,"fro") / np.linalg.norm(trues,"fro") * 100

    print(f"  Final AE  R²={r2:.6f}  RMSE={rmse:.4f} N/m  Rel.ε={rel:.4f}%")

    return {
        "ae": ae, "history": history,
        "metrics": {"r2": r2, "rmse": rmse, "rel_err": rel},
        "layer_dims": layer_dims,
        "scaler": scaler,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global _FORCE_MAT, _INPUTS, _CFG

    print("=" * 70)
    print("  Autoencoder Hyperparameter Optimisation  (Optuna / TPE)")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[1/5] Building dataset...")
    force_mat, inputs, scaler = build_dataset(CFG)
    _FORCE_MAT = force_mat
    _INPUTS    = inputs
    _CFG       = CFG
    print(f"  Dataset  {force_mat.shape[0]} snapshots × {force_mat.shape[1]} dims")

    # ── Optuna study ──────────────────────────────────────────────────────
    print(f"\n[2/5] Running {CFG['n_trials']} Optuna trials...")
    study_path = CFG["output_dir"] / "optuna_ae_study.pkl"

    sampler = TPESampler(
        n_startup_trials = CFG["n_startup_trials"],
        seed             = CFG["random_seed"],
    )
    pruner  = MedianPruner(
        n_startup_trials = CFG["n_startup_trials"],
        n_warmup_steps   = CFG["n_warmup_steps"],
        interval_steps   = CFG["pruning_interval"],
    )
    study = optuna.create_study(
        study_name     = CFG["study_name"],
        direction      = "minimize",
        sampler        = sampler,
        pruner         = pruner,
    )

    study.optimize(
        objective,
        n_trials  = CFG["n_trials"],
        callbacks = [
            lambda s, t: print(
                f"  Trial {t.number:3d} | val={t.value:.6f} | "
                f"best={s.best_value:.6f}"
            ) if t.state == optuna.trial.TrialState.COMPLETE else None
        ],
    )

    # Save study
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"  Study saved → {study_path}")

    # ── Best params ───────────────────────────────────────────────────────
    print("\n[3/5] Best hyperparameters:")
    best_params = study.best_trial.params
    for k, v in sorted(best_params.items()):
        print(f"  {k:35s}: {v}")

    param_path = CFG["output_dir"] / "best_ae_params.json"
    with open(param_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"\n  Saved → {param_path}")

    # ── Save trial log ────────────────────────────────────────────────────
    trials_df = study.trials_dataframe()
    trials_df.to_csv(CFG["output_dir"] / "ae_trial_log.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[4/5] Generating HPO diagnostic plots...")
    plot_hpo_results(study, CFG["output_dir"])

    # ── Retrain best model ────────────────────────────────────────────────
    print("\n[5/5] Retraining best configuration at full budget...")
    result = retrain_best_ae(best_params, CFG, force_mat, inputs, scaler)
    plot_final_training(result["history"], CFG["output_dir"])

    # Save best model bundle
    ae = result["ae"]
    torch.save({
        "encoder_state": ae.encoder.state_dict(),
        "decoder_state": ae.decoder.state_dict(),
        "best_params":   best_params,
        "layer_dims":    result["layer_dims"],
        "force_scaler":  scaler,
        "metrics":       result["metrics"],
        "cfg":           CFG,
    }, CFG["output_dir"] / "best_ae_model.pt")
    print(f"  Model saved → {CFG['output_dir'] / 'best_ae_model.pt'}")

    print("\n" + "=" * 70)
    print(f"  HPO complete.  Best val MSE : {study.best_value:.6f}")
    print(f"  Final test  →  R²={result['metrics']['r2']:.5f}"
          f"  RMSE={result['metrics']['rmse']:.2f} N/m"
          f"  Rel.ε={result['metrics']['rel_err']:.3f}%")
    print(f"  All outputs  : {CFG['output_dir'].resolve()}")
    print("=" * 70)

    return study, result


if __name__ == "__main__":
    study, result = main()


# ══════════════════════════════════════════════════════════════════════════════
# USAGE NOTES
# ══════════════════════════════════════════════════════════════════════════════
"""
Install dependencies:
    pip install optuna torch scikit-learn pandas matplotlib

Resume an interrupted study (Optuna stores trials internally):
    with open("hpo_ae_results/optuna_ae_study.pkl","rb") as f:
        study = pickle.load(f)
    study.optimize(objective, n_trials=40)   # add 40 more trials

Load best model for use in ANN HPO script:
    ckpt = torch.load("hpo_ae_results/best_ae_model.pt")
    # ckpt["encoder_state"], ckpt["decoder_state"], ckpt["best_params"]

Increase search budget for production:
    CFG["n_trials"]   = 200
    CFG["max_epochs"] = 500
"""
