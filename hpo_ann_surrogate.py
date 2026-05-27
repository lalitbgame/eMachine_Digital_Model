"""
================================================================================
  Script 2 — Hyperparameter Optimisation: ANN Surrogate Model
  ─────────────────────────────────────────────────────────────────────────────
  Framework  : Optuna (TPE sampler + MedianPruner)
  Objective  : minimise validation MSE in PHYSICAL force space
               (ANN predicts ẑ → frozen Decoder → x̂, loss = MSE(x̂, x))

  Prerequisites
  ─────────────
  Run hpo_autoencoder.py first. This script loads:
    hpo_ae_results/best_ae_model.pt   ← frozen decoder weights + force scaler

  Search space
  ────────────
  Architecture
    • n_hidden_layers      : 2 – 6
    • hidden_dim per layer : 32 – 512  (log-uniform, can grow then shrink)
    • topology             : monotone_decrease | bottleneck | free
    • activation           : GELU | ReLU | ELU | Tanh | SiLU
    • use_skip_connection  : True | False  (residual from input to output)
    • use_batch_norm       : True | False
    • use_layer_norm       : True | False

  Input encoding
    • angle_encoding       : raw | sincos | fourier_k
      - raw    : [RPM, T, θ_deg]          (3 features)
      - sincos : [RPM, T, sin θ, cos θ]   (4 features; removes 360/0 discontinuity)
      - fourier: [RPM, T, sin kθ, cos kθ for k=1..K] (2K+2 features)
    • fourier_k            : 2 – 8  (only if fourier encoding chosen)

  Regularisation
    • dropout              : 0.0 – 0.4
    • weight_decay         : 1e-6 – 1e-3
    • gradient_clip        : 0.5 – 5.0

  Optimiser & schedule
    • optimizer            : AdamW | Adam | RMSprop
    • learning_rate        : 1e-4 – 5e-3
    • lr_scheduler         : CosineAnnealing | ReduceLROnPlateau | StepLR | None
    • batch_size           : 64, 128, 256, 512

  Loss strategy
    • loss_mode            : end_to_end | latent
      end_to_end: MSE(Decoder(ẑ), x)   — preferred, keeps physical meaning
      latent    : MSE(ẑ, z_encoder(x)) — faster, decoupled

  Outputs
  ───────
  hpo_ann_results/
    ├── best_ann_params.json            best trial hyperparameters
    ├── best_ann_model.pt               ANN + decoder + scalers (inference-ready)
    ├── optuna_ann_study.pkl            full Optuna study (resumable)
    ├── ann_hpo_history.png             optimisation history
    ├── ann_param_importance.png        parameter importances
    ├── ann_parallel_coords.png         parallel coordinates
    ├── ann_parity_plot.png             predicted vs true forces
    ├── ann_error_by_torque.png         per-torque error breakdown
    └── ann_trial_log.csv               all trial results
================================================================================
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # Data
    "n_teeth":          48,
    "input_dim":        96,
    "speed_rpm":        1000,
    "torque_values":    [10, 20, 30, 40, 50, 60, 70, 80, 100, 120],
    "n_angles":         360,
    "data_path":        None,

    # Path to frozen decoder from AE HPO (script 1 output)
    "ae_model_path":    Path("hpo_ae_results/best_ae_model.pt"),

    # Optuna study
    "n_trials":         100,
    "n_startup_trials": 20,
    "n_warmup_steps":   15,
    "pruning_interval": 5,

    # Per-trial budget
    "max_epochs":       300,
    "early_stop_patience": 30,

    # Final retraining
    "final_epochs":     600,
    "final_patience":   60,

    # Splits
    "train_split":      0.75,
    "val_split":        0.15,
    "random_seed":      42,

    "output_dir":       Path("hpo_ann_results"),
    "study_name":       "emag_ann_surrogate_hpo",
}

CFG["output_dir"].mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(CFG["random_seed"])
np.random.seed(CFG["random_seed"])
print(f"Device : {DEVICE}")
print(f"Trials : {CFG['n_trials']}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset(cfg):
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


def make_splits(force_mat, inputs, cfg):
    n    = len(force_mat)
    n_tr = int(n * cfg["train_split"])
    n_va = int(n * cfg["val_split"])
    n_te = n - n_tr - n_va
    ds   = TensorDataset(torch.tensor(force_mat), torch.tensor(inputs))
    g    = torch.Generator().manual_seed(cfg["random_seed"])
    return random_split(ds, [n_tr, n_va, n_te], generator=g)


def make_loader(ds, batch_size, shuffle=True):
    pin = DEVICE.type == "cuda"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin, drop_last=shuffle)


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD FROZEN DECODER FROM AE HPO
# ══════════════════════════════════════════════════════════════════════════════

# These classes must match hpo_autoencoder.py exactly
def _make_norm(norm_type, dim):
    if norm_type == "batch":  return nn.BatchNorm1d(dim)
    if norm_type == "layer":  return nn.LayerNorm(dim)
    return None

def _get_act_cls(name):
    return {"GELU": nn.GELU, "ReLU": nn.ReLU, "ELU": nn.ELU,
            "Tanh": nn.Tanh}[name]

class ResidualBlock(nn.Module):
    def __init__(self, in_d, out_d, norm_type, act_cls, dropout):
        super().__init__()
        self.linear = nn.Linear(in_d, out_d)
        norm = _make_norm(norm_type, out_d)
        self.norm = norm if norm else nn.Identity()
        self.act  = act_cls()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_d, out_d, bias=False) if in_d != out_d else nn.Identity()
    def forward(self, x):
        return self.act(self.norm(self.linear(x))) + self.proj(x)

class FlexDecoder(nn.Module):
    def __init__(self, latent_dim, layer_dims, output_dim,
                 norm_type, act_name, dropout, use_skip):
        super().__init__()
        act_cls = _get_act_cls(act_name)
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


def load_frozen_decoder(ae_model_path: Path, cfg: dict):
    """Load best_ae_model.pt from HPO script 1, return frozen decoder."""
    if not ae_model_path.exists():
        print(f"  [WARN] AE model not found at {ae_model_path}.")
        print("  [WARN] Using a dummy decoder (latent_dim=16 → 96 linear).")
        dec = nn.Linear(16, cfg["input_dim"]).to(DEVICE)
        for p in dec.parameters():
            p.requires_grad = False
        return dec, 16, None

    ckpt       = torch.load(ae_model_path, map_location=DEVICE)
    bp         = ckpt["best_params"]
    layer_dims = ckpt["layer_dims"]
    latent_dim = bp["latent_dim"]

    decoder = FlexDecoder(
        latent_dim  = latent_dim,
        layer_dims  = list(reversed(layer_dims)),
        output_dim  = cfg["input_dim"],
        norm_type   = bp["norm_type"],
        act_name    = bp["activation"],
        dropout     = 0.0,          # no dropout at inference / surrogate training
        use_skip    = bp["use_skip"],
    ).to(DEVICE)
    decoder.load_state_dict(ckpt["decoder_state"])
    for p in decoder.parameters():
        p.requires_grad = False
    decoder.eval()

    force_scaler = ckpt.get("force_scaler", None)
    print(f"  Frozen decoder loaded  latent_dim={latent_dim}  "
          f"layer_dims={list(reversed(layer_dims))}")
    return decoder, latent_dim, force_scaler


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANGLE ENCODING
# ══════════════════════════════════════════════════════════════════════════════
def encode_conditions(inputs_raw: np.ndarray,
                      encoding: str,
                      fourier_k: int = 4,
                      cond_scaler: StandardScaler = None,
                      fit_scaler: bool = False) -> tuple:
    """
    inputs_raw : [N × 3]  [rpm, torque, angle_deg]
    Returns (encoded [N × D], cond_scaler, D)

    encoding options
    ────────────────
    'raw'     → standardise [rpm, torque, angle_deg]            D = 3
    'sincos'  → standardise [rpm, torque] + [sin θ, cos θ]     D = 4
    'fourier' → standardise [rpm, torque] + [sin kθ, cos kθ]   D = 2+2K
    """
    rpm    = inputs_raw[:, 0:1]
    torque = inputs_raw[:, 1:2]
    angle  = inputs_raw[:, 2:3]   # degrees
    theta  = np.deg2rad(angle)

    if encoding == "raw":
        features = np.hstack([rpm, torque, angle])

    elif encoding == "sincos":
        features = np.hstack([rpm, torque,
                               np.sin(theta), np.cos(theta)])

    elif encoding == "fourier":
        harmonics = [np.sin((k+1)*theta) for k in range(fourier_k)] + \
                    [np.cos((k+1)*theta) for k in range(fourier_k)]
        features  = np.hstack([rpm, torque] + harmonics)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    if fit_scaler:
        cond_scaler = StandardScaler()
        # For sincos/fourier features the scaler is still useful for rpm/torque
        # but we only scale the first 2 dims; angle features are already bounded.
        if encoding in ("sincos", "fourier"):
            scaler_part = StandardScaler()
            scaler_part.fit(features[:, :2])
            scaled_meta = scaler_part.transform(features[:, :2])
            features    = np.hstack([scaled_meta, features[:, 2:]])
            cond_scaler = scaler_part   # store partial scaler
        else:
            features = cond_scaler.fit_transform(features)
    else:
        if cond_scaler is not None:
            if encoding in ("sincos", "fourier"):
                scaled_meta = cond_scaler.transform(features[:, :2])
                features    = np.hstack([scaled_meta, features[:, 2:]])
            else:
                features    = cond_scaler.transform(features)

    return features.astype(np.float32), cond_scaler, features.shape[1]


# ══════════════════════════════════════════════════════════════════════════════
# 4. FLEXIBLE ANN SURROGATE
# ══════════════════════════════════════════════════════════════════════════════
def _get_act(name: str) -> nn.Module:
    return {"GELU": nn.GELU, "ReLU": nn.ReLU, "ELU": nn.ELU,
            "Tanh": nn.Tanh, "SiLU": nn.SiLU}[name]()


class FlexANN(nn.Module):
    """
    Flexible MLP surrogate with three topology options:
      monotone_decrease : dims strictly decrease from in_dim → latent_dim
      bottleneck        : dims widen then narrow  (hourglass)
      free              : each layer dim sampled independently
    Optional linear skip from input → output.
    """
    def __init__(self, in_dim: int, hidden_dims: list, out_dim: int,
                 act_name: str, norm_type: str, dropout: float,
                 use_skip: bool):
        super().__init__()
        act_cls = _get_act(act_name).__class__
        layers  = []
        prev    = in_dim
        for d in hidden_dims:
            blk = [nn.Linear(prev, d)]
            if norm_type == "batch": blk.append(nn.BatchNorm1d(d))
            if norm_type == "layer": blk.append(nn.LayerNorm(d))
            blk += [act_cls(), nn.Dropout(dropout)]
            layers.append(nn.Sequential(*blk))
            prev = d
        self.hidden = nn.ModuleList(layers)
        self.out    = nn.Linear(prev, out_dim)
        self.skip   = nn.Linear(in_dim, out_dim, bias=False) if use_skip else None

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = layer(h)
        out = self.out(h)
        if self.skip is not None:
            out = out + self.skip(x)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD ANN FROM TRIAL
# ══════════════════════════════════════════════════════════════════════════════
def build_ann_from_trial(trial: optuna.Trial,
                         in_dim: int,
                         latent_dim: int) -> FlexANN:
    """Sample and build an ANN from an Optuna trial."""

    topology   = trial.suggest_categorical(
        "topology", ["monotone_decrease", "bottleneck", "free"])
    n_layers   = trial.suggest_int("n_hidden_layers", 2, 6)
    act_name   = trial.suggest_categorical(
        "activation", ["GELU", "ReLU", "ELU", "Tanh", "SiLU"])
    norm_type  = trial.suggest_categorical(
        "norm_type", ["batch", "layer", "none"])
    use_skip   = trial.suggest_categorical("use_skip", [True, False])
    dropout    = trial.suggest_float("dropout", 0.0, 0.4)

    # ── Layer dims by topology ────────────────────────────────────────────
    if topology == "monotone_decrease":
        # Start wide, shrink toward latent_dim
        first = trial.suggest_int("first_dim", max(latent_dim*2, 64), 512, log=True)
        hidden_dims = np.linspace(first, latent_dim*2, n_layers, dtype=int).tolist()

    elif topology == "bottleneck":
        # Wide middle, narrow on both sides
        peak    = trial.suggest_int("peak_dim", max(latent_dim*2, 64), 512, log=True)
        valley  = trial.suggest_int("valley_dim", max(latent_dim, 32),
                                     max(peak//2, latent_dim+1), log=True)
        half    = n_layers // 2
        up   = np.linspace(valley, peak, half+1, dtype=int)[1:].tolist()
        down = np.linspace(peak, valley, n_layers - half + 1, dtype=int)[1:].tolist()
        hidden_dims = up + down

    else:   # free
        hidden_dims = [
            trial.suggest_int(f"layer_{i}_dim", max(latent_dim, 32), 512, log=True)
            for i in range(n_layers)
        ]

    return FlexANN(
        in_dim      = in_dim,
        hidden_dims = hidden_dims,
        out_dim     = latent_dim,
        act_name    = act_name,
        norm_type   = norm_type,
        dropout     = dropout,
        use_skip    = use_skip,
    ).to(DEVICE)


def build_optimizer(trial, params):
    opt_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "RMSprop"])
    lr       = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    wd       = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    if opt_name == "AdamW":   return optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "Adam":  return optim.Adam(params, lr=lr, weight_decay=wd)
    else:                     return optim.RMSprop(params, lr=lr, weight_decay=wd)


def build_scheduler(trial, optimizer, max_epochs):
    name = trial.suggest_categorical(
        "lr_scheduler",
        ["CosineAnnealing", "ReduceLROnPlateau", "StepLR", "None"])
    if name == "CosineAnnealing":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6), name
    elif name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-6), name
    elif name == "StepLR":
        step  = trial.suggest_int("steplr_step", 20, 100)
        gamma = trial.suggest_float("steplr_gamma", 0.3, 0.9)
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step, gamma=gamma), name
    else:
        return None, "None"


# ══════════════════════════════════════════════════════════════════════════════
# 6. LATENT CODE EXTRACTION  (for 'latent' loss mode)
# ══════════════════════════════════════════════════════════════════════════════
class FlexEncoder(nn.Module):
    """Mirror of FlexDecoder — needed to load encoder state from AE HPO."""
    def __init__(self, input_dim, layer_dims, latent_dim,
                 norm_type, act_name, dropout, use_skip):
        super().__init__()
        act_cls = _get_act_cls(act_name)
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


@torch.no_grad()
def extract_latent_codes(ae_model_path: Path, force_mat: np.ndarray,
                         cfg: dict) -> np.ndarray:
    """Run the saved encoder on force_mat, return z [N × latent_dim]."""
    if not ae_model_path.exists():
        print("  [WARN] No encoder found — returning random latent codes.")
        ld = 16
        return np.random.randn(len(force_mat), ld).astype(np.float32)

    ckpt       = torch.load(ae_model_path, map_location=DEVICE)
    bp         = ckpt["best_params"]
    layer_dims = ckpt["layer_dims"]
    latent_dim = bp["latent_dim"]

    encoder = FlexEncoder(
        input_dim  = cfg["input_dim"],
        layer_dims = layer_dims,
        latent_dim = latent_dim,
        norm_type  = bp["norm_type"],
        act_name   = bp["activation"],
        dropout    = 0.0,
        use_skip   = bp["use_skip"],
    ).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    x_t = torch.tensor(force_mat)
    ld  = DataLoader(TensorDataset(x_t), batch_size=512)
    zs  = [encoder(xb[0].to(DEVICE)).cpu().numpy() for xb in ld]
    return np.vstack(zs)


# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train_ann_loop(ann, decoder, optimizer, scheduler, sched_name,
                   tr_ld, va_ld,
                   loss_mode: str,
                   latent_codes_train=None,
                   latent_codes_val=None,
                   max_epochs=300, patience=30, clip_val=1.0,
                   trial=None, pruning_interval=5):
    """
    loss_mode == 'end_to_end': ANN → ẑ → Decoder → MSE(x̂, x)
    loss_mode == 'latent'    : ANN → ẑ → MSE(ẑ, z_enc)
    """
    crit    = nn.MSELoss()
    best_val, best_sd = np.inf, None
    no_imp  = 0
    history = {"train": [], "val": []}

    # For latent mode, build loaders from pre-extracted codes
    if loss_mode == "latent":
        assert latent_codes_train is not None and latent_codes_val is not None
        # tr_ld and va_ld are replaced by (cond, z_target) loaders
        # (already passed in by the objective)

    for epoch in range(1, max_epochs + 1):
        # ── Train ──
        ann.train()
        tr_loss = 0.0
        for batch in tr_ld:
            if loss_mode == "end_to_end":
                x_b, c_b = batch
                x_b = x_b.to(DEVICE);  c_b = c_b.to(DEVICE)
                optimizer.zero_grad()
                z_hat = ann(c_b)
                x_hat = decoder(z_hat)
                loss  = crit(x_hat, x_b)
            else:   # latent
                c_b, z_b = batch
                c_b = c_b.to(DEVICE);  z_b = z_b.to(DEVICE)
                optimizer.zero_grad()
                z_hat = ann(c_b)
                loss  = crit(z_hat, z_b)

            loss.backward()
            nn.utils.clip_grad_norm_(ann.parameters(), clip_val)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        # ── Validate ──
        ann.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in va_ld:
                if loss_mode == "end_to_end":
                    x_b, c_b = batch
                    z_hat = ann(c_b.to(DEVICE))
                    x_hat = decoder(z_hat)
                    va_loss += crit(x_hat, x_b.to(DEVICE)).item()
                else:
                    c_b, z_b = batch
                    va_loss += crit(ann(c_b.to(DEVICE)),
                                    z_b.to(DEVICE)).item()
        va_loss /= len(va_ld)

        if scheduler is not None:
            scheduler.step(va_loss) if sched_name == "ReduceLROnPlateau" \
                else scheduler.step()

        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_sd  = deepcopy(ann.state_dict())
            no_imp   = 0
        else:
            no_imp += 1

        if trial is not None and epoch % pruning_interval == 0:
            trial.report(va_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if no_imp >= patience:
            break

    ann.load_state_dict(best_sd)
    return best_val, history


# ══════════════════════════════════════════════════════════════════════════════
# 8. OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════
_FORCE_MAT    = None
_INPUTS       = None
_CFG          = None
_DECODER      = None
_LATENT_DIM   = None
_TRAIN_DS     = None
_VAL_DS       = None
_LATENT_CODES = None   # pre-extracted [N × d]


def objective(trial: optuna.Trial) -> float:
    cfg         = _CFG
    loss_mode   = trial.suggest_categorical("loss_mode",
                                             ["end_to_end", "latent"])
    angle_enc   = trial.suggest_categorical("angle_encoding",
                                             ["raw", "sincos", "fourier"])
    fourier_k   = trial.suggest_int("fourier_k", 2, 8) \
                  if angle_enc == "fourier" else 4
    batch_size  = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    clip_val    = trial.suggest_float("grad_clip", 0.5, 5.0)

    # ── Encode conditions ─────────────────────────────────────────────────
    all_inputs  = torch.cat([
        c for _, c in DataLoader(_TRAIN_DS, batch_size=len(_TRAIN_DS))
    ]).numpy()
    all_val_inp = torch.cat([
        c for _, c in DataLoader(_VAL_DS,   batch_size=len(_VAL_DS))
    ]).numpy()

    enc_tr, cond_sc, in_dim = encode_conditions(
        all_inputs, angle_enc, fourier_k, fit_scaler=True)
    enc_va, _, _            = encode_conditions(
        all_val_inp, angle_enc, fourier_k, cond_sc)

    enc_tr_t = torch.tensor(enc_tr)
    enc_va_t = torch.tensor(enc_va)

    # ── Build loaders ─────────────────────────────────────────────────────
    if loss_mode == "end_to_end":
        all_x_tr = torch.cat([x for x, _ in DataLoader(_TRAIN_DS, batch_size=len(_TRAIN_DS))])
        all_x_va = torch.cat([x for x, _ in DataLoader(_VAL_DS,   batch_size=len(_VAL_DS))])
        tr_ld = DataLoader(TensorDataset(all_x_tr, enc_tr_t),
                           batch_size=batch_size, shuffle=True, drop_last=True)
        va_ld = DataLoader(TensorDataset(all_x_va, enc_va_t),
                           batch_size=512)
    else:   # latent: need (cond, z) pairs
        n_tr  = len(all_inputs)
        n_va  = len(all_val_inp)
        # Find matching indices in full dataset (train/val share same seed)
        z_tr  = _LATENT_CODES[:n_tr]
        z_va  = _LATENT_CODES[n_tr:n_tr+n_va]
        tr_ld = DataLoader(TensorDataset(enc_tr_t, torch.tensor(z_tr)),
                           batch_size=batch_size, shuffle=True, drop_last=True)
        va_ld = DataLoader(TensorDataset(enc_va_t, torch.tensor(z_va)),
                           batch_size=512)

    # ── Build model ───────────────────────────────────────────────────────
    ann       = build_ann_from_trial(trial, in_dim, _LATENT_DIM)
    optimizer = build_optimizer(trial, ann.parameters())
    scheduler, sched_name = build_scheduler(trial, optimizer, cfg["max_epochs"])

    best_val, _ = train_ann_loop(
        ann, _DECODER, optimizer, scheduler, sched_name,
        tr_ld, va_ld,
        loss_mode        = loss_mode,
        max_epochs       = cfg["max_epochs"],
        patience         = cfg["early_stop_patience"],
        clip_val         = clip_val,
        trial            = trial,
        pruning_interval = cfg["pruning_interval"],
    )
    return best_val


# ══════════════════════════════════════════════════════════════════════════════
# 9. FINAL RETRAINING
# ══════════════════════════════════════════════════════════════════════════════
def retrain_best_ann(best_params: dict, cfg: dict,
                     decoder, latent_dim: int,
                     force_mat, inputs,
                     force_scaler: StandardScaler) -> dict:
    """Rebuild best ANN and retrain on train+val, evaluate on test."""
    print("\n── Retraining best ANN surrogate at full budget ───────────────")

    # ── Encode all conditions ─────────────────────────────────────────────
    angle_enc = best_params.get("angle_encoding", "sincos")
    fourier_k = best_params.get("fourier_k", 4)
    enc_all, cond_scaler, in_dim = encode_conditions(
        inputs, angle_enc, fourier_k, fit_scaler=True)

    # ── Rebuild hidden dims from topology ─────────────────────────────────
    topology = best_params["topology"]
    n_layers = best_params["n_hidden_layers"]

    if topology == "monotone_decrease":
        first = best_params["first_dim"]
        hidden_dims = np.linspace(first, latent_dim*2, n_layers, dtype=int).tolist()
    elif topology == "bottleneck":
        peak   = best_params["peak_dim"]
        valley = best_params["valley_dim"]
        half   = n_layers // 2
        up   = np.linspace(valley, peak, half+1, dtype=int)[1:].tolist()
        down = np.linspace(peak, valley, n_layers-half+1, dtype=int)[1:].tolist()
        hidden_dims = up + down
    else:
        hidden_dims = [best_params[f"layer_{i}_dim"] for i in range(n_layers)]

    ann = FlexANN(
        in_dim      = in_dim,
        hidden_dims = hidden_dims,
        out_dim     = latent_dim,
        act_name    = best_params["activation"],
        norm_type   = best_params["norm_type"],
        dropout     = best_params["dropout"],
        use_skip    = best_params["use_skip"],
    ).to(DEVICE)

    # ── Splits: use train+val for final training ──────────────────────────
    n     = len(force_mat)
    n_tr  = int(n * cfg["train_split"])
    n_va  = int(n * cfg["val_split"])
    n_te  = n - n_tr - n_va
    n_trva = n_tr + n_va

    x_t   = torch.tensor(force_mat)
    enc_t = torch.tensor(enc_all)
    g     = torch.Generator().manual_seed(cfg["random_seed"])
    trva_ds, te_ds = random_split(
        TensorDataset(x_t, enc_t), [n_trva, n_te], generator=g)

    loss_mode = best_params.get("loss_mode", "end_to_end")
    bs        = best_params["batch_size"]

    if loss_mode == "end_to_end":
        tr_ld = DataLoader(trva_ds, batch_size=bs, shuffle=True, drop_last=True)
        va_ld = DataLoader(te_ds,   batch_size=512)
    else:
        # latent mode: need (enc_cond, z_target)
        z_all  = extract_latent_codes(cfg["ae_model_path"], force_mat, cfg)
        z_t    = torch.tensor(z_all)
        g2     = torch.Generator().manual_seed(cfg["random_seed"])
        trva2, te2 = random_split(
            TensorDataset(enc_t, z_t), [n_trva, n_te], generator=g2)
        tr_ld = DataLoader(trva2, batch_size=bs, shuffle=True, drop_last=True)
        va_ld = DataLoader(te2,   batch_size=512)

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    lr  = best_params["lr"]
    wd  = best_params["weight_decay"]
    oname = best_params["optimizer"]
    if oname == "AdamW":   optimizer = optim.AdamW(ann.parameters(), lr=lr, weight_decay=wd)
    elif oname == "Adam":  optimizer = optim.Adam(ann.parameters(), lr=lr, weight_decay=wd)
    else:                  optimizer = optim.RMSprop(ann.parameters(), lr=lr, weight_decay=wd)

    sname = best_params["lr_scheduler"]
    if sname == "CosineAnnealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["final_epochs"], eta_min=1e-6)
    elif sname == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, factor=0.5)
    elif sname == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size = best_params.get("steplr_step", 50),
            gamma     = best_params.get("steplr_gamma", 0.5))
    else:
        scheduler = None

    best_val, history = train_ann_loop(
        ann, decoder, optimizer, scheduler, sname,
        tr_ld, va_ld,
        loss_mode  = loss_mode,
        max_epochs = cfg["final_epochs"],
        patience   = cfg["final_patience"],
        clip_val   = best_params["grad_clip"],
    )

    # ── Full-pipeline evaluation on test set ──────────────────────────────
    ann.eval(); decoder.eval()
    preds, trues, conds = [], [], []
    with torch.no_grad():
        for batch in va_ld:
            if loss_mode == "end_to_end":
                x_b, c_b = batch
                z_hat = ann(c_b.to(DEVICE))
                x_hat = decoder(z_hat)
                preds.append(x_hat.cpu().numpy())
                trues.append(x_b.numpy())
                conds.append(c_b.numpy())
            else:
                c_b, z_b = batch
                z_hat = ann(c_b.to(DEVICE))
                x_hat = decoder(z_hat)
                preds.append(x_hat.cpu().numpy())
                # For evaluation we need x_true — get from te_ds indices
                conds.append(c_b.numpy())

    preds_sc = np.vstack(preds)

    if loss_mode == "end_to_end":
        trues_sc = np.vstack(trues)
    else:
        # Need original x for test set
        te_indices = te_ds.indices if hasattr(te_ds, 'indices') else list(range(len(te_ds)))
        trues_sc   = force_mat[te_indices[:len(preds_sc)]]

    if force_scaler is not None:
        preds_phy = force_scaler.inverse_transform(preds_sc)
        trues_phy = force_scaler.inverse_transform(trues_sc)
    else:
        preds_phy = preds_sc
        trues_phy = trues_sc

    r2   = r2_score(trues_phy.ravel(), preds_phy.ravel())
    rmse = np.sqrt(mean_squared_error(trues_phy.ravel(), preds_phy.ravel()))
    rel  = (np.linalg.norm(trues_phy-preds_phy, "fro") /
            np.linalg.norm(trues_phy, "fro") * 100)
    n_t = cfg["n_teeth"]
    r2_fr = r2_score(trues_phy[:, :n_t].ravel(), preds_phy[:, :n_t].ravel())
    r2_ft = r2_score(trues_phy[:, n_t:].ravel(), preds_phy[:, n_t:].ravel())

    print(f"  Test  R²={r2:.6f}  R²_Fr={r2_fr:.6f}  R²_Ft={r2_ft:.6f}")
    print(f"        RMSE={rmse:.4f} N/m  Rel.ε={rel:.4f}%")

    return {
        "ann": ann, "history": history,
        "cond_scaler": cond_scaler,
        "hidden_dims": hidden_dims,
        "angle_encoding": angle_enc,
        "fourier_k": fourier_k,
        "metrics": {"r2": r2, "r2_fr": r2_fr, "r2_ft": r2_ft,
                    "rmse": rmse, "rel_err": rel},
        "preds_phy": preds_phy,
        "trues_phy": trues_phy,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
def plot_hpo_results(study: optuna.Study, out_dir: Path):
    trials_df  = study.trials_dataframe(attrs=("number","value","state","duration"))
    completed  = trials_df[trials_df["state"] == "COMPLETE"]

    # 1. History
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(completed["number"], completed["value"],
               s=18, alpha=0.6, color="#185FA5", label="Trial val loss")
    ax.plot(completed["number"], completed["value"].cummin(),
            color="#D85A30", lw=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("Val MSE loss")
    ax.set_title("ANN Surrogate HPO — optimisation history")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "ann_hpo_history.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Parameter importances
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(9, max(4, len(importances)*0.4)))
        params  = list(importances.keys())
        vals    = list(importances.values())
        colors  = ["#1D9E75" if v == max(vals) else "#534AB7" for v in vals]
        ax.barh(params, vals, color=colors)
        ax.set_xlabel("Importance score")
        ax.set_title("ANN Surrogate HPO — parameter importances (fANOVA)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "ann_param_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    # 3. Best params card
    fig, ax = plt.subplots(figsize=(9, 5))
    best    = study.best_trial
    ax.set_title(f"Best trial #{best.number}  val={best.value:.6f}", fontsize=12)
    lines   = [f"{k:35s}: {v}" for k, v in sorted(best.params.items())]
    ax.text(0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="#F1EFE8", alpha=0.7))
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "ann_best_trial_params.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  HPO plots saved to {out_dir}")


def plot_final_results(result: dict, cfg: dict, out_dir: Path):
    n       = cfg["n_teeth"]
    history = result["history"]
    preds   = result["preds_phy"]
    trues   = result["trues_phy"]

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

    # ── Training curves ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ep = np.arange(1, len(history["train"])+1)
    ax.semilogy(ep, history["train"], label="Train", color="#1D9E75")
    ax.semilogy(ep, history["val"],   label="Val",   color="#D85A30")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("Final ANN training"); ax.legend(); ax.grid(alpha=0.3)

    # ── Parity Fr ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(trues[:, :n].ravel(), preds[:, :n].ravel(),
               s=1, alpha=0.3, color="#185FA5")
    lim = [trues[:, :n].min(), trues[:, :n].max()]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("FEA Fr [N/m]"); ax.set_ylabel("Predicted Fr [N/m]")
    r2_fr = result["metrics"]["r2_fr"]
    ax.set_title(f"Parity Fr  R²={r2_fr:.5f}"); ax.grid(alpha=0.3)

    # ── Parity Ft ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(trues[:, n:].ravel(), preds[:, n:].ravel(),
               s=1, alpha=0.3, color="#1D9E75")
    lim = [trues[:, n:].min(), trues[:, n:].max()]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("FEA Ft [N/m]"); ax.set_ylabel("Predicted Ft [N/m]")
    r2_ft = result["metrics"]["r2_ft"]
    ax.set_title(f"Parity Ft  R²={r2_ft:.5f}"); ax.grid(alpha=0.3)

    # ── Per-tooth RMSE Fr ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    rmse_per_tooth = np.sqrt(((trues[:, :n] - preds[:, :n])**2).mean(axis=0))
    ax.bar(np.arange(n), rmse_per_tooth, color="#185FA5", alpha=0.8)
    ax.set_xlabel("Tooth index"); ax.set_ylabel("RMSE [N/m]")
    ax.set_title("Per-tooth RMSE — Fr"); ax.grid(alpha=0.3)

    # ── Per-tooth RMSE Ft ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    rmse_per_tooth_ft = np.sqrt(((trues[:, n:] - preds[:, n:])**2).mean(axis=0))
    ax.bar(np.arange(n), rmse_per_tooth_ft, color="#1D9E75", alpha=0.8)
    ax.set_xlabel("Tooth index"); ax.set_ylabel("RMSE [N/m]")
    ax.set_title("Per-tooth RMSE — Ft"); ax.grid(alpha=0.3)

    # ── Single snapshot overlay ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    snap = 0
    ax.plot(np.arange(n), trues[snap, :n], "o-", ms=3,
            label="FEA Fr", color="#185FA5", lw=1.5)
    ax.plot(np.arange(n), preds[snap, :n], "s--", ms=3,
            label="ANN Fr", color="#D85A30", lw=1.4)
    ax.set_xlabel("Tooth index"); ax.set_ylabel("Fr [N/m]")
    ax.set_title(f"Snapshot {snap} — Fr per tooth")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    m = result["metrics"]
    fig.suptitle(
        f"ANN Surrogate — Final test  R²={m['r2']:.5f}  "
        f"RMSE={m['rmse']:.2f} N/m  Rel.ε={m['rel_err']:.3f}%",
        fontsize=13, fontweight="bold")

    fig.savefig(out_dir / "ann_final_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir / 'ann_final_results.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global _FORCE_MAT, _INPUTS, _CFG, _DECODER, _LATENT_DIM
    global _TRAIN_DS, _VAL_DS, _LATENT_CODES

    print("=" * 70)
    print("  ANN Surrogate Hyperparameter Optimisation  (Optuna / TPE)")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[1/6] Building dataset...")
    force_mat, inputs, force_scaler = build_dataset(CFG)
    _FORCE_MAT = force_mat
    _INPUTS    = inputs
    _CFG       = CFG
    print(f"  Dataset {force_mat.shape[0]} snapshots × {force_mat.shape[1]} dims")

    # ── Frozen decoder ────────────────────────────────────────────────────
    print("\n[2/6] Loading frozen decoder from AE HPO results...")
    decoder, latent_dim, ae_force_scaler = load_frozen_decoder(
        CFG["ae_model_path"], CFG)
    _DECODER    = decoder
    _LATENT_DIM = latent_dim

    # If AE HPO provided its own scaler, prefer that for consistency
    if ae_force_scaler is not None:
        force_scaler = ae_force_scaler

    # ── Dataset splits ────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = make_splits(force_mat, inputs, CFG)
    _TRAIN_DS = train_ds
    _VAL_DS   = val_ds

    # Pre-extract latent codes for 'latent' loss mode trials
    print("  Extracting latent codes (for latent-loss trials)...")
    _LATENT_CODES = extract_latent_codes(CFG["ae_model_path"], force_mat, CFG)

    # ── Optuna ────────────────────────────────────────────────────────────
    print(f"\n[3/6] Running {CFG['n_trials']} Optuna trials...")
    study_path = CFG["output_dir"] / "optuna_ann_study.pkl"

    study = optuna.create_study(
        study_name = CFG["study_name"],
        direction  = "minimize",
        sampler    = TPESampler(n_startup_trials=CFG["n_startup_trials"],
                                seed=CFG["random_seed"]),
        pruner     = MedianPruner(n_startup_trials=CFG["n_startup_trials"],
                                  n_warmup_steps=CFG["n_warmup_steps"]),
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

    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"  Study saved → {study_path}")

    # ── Best params ───────────────────────────────────────────────────────
    print("\n[4/6] Best hyperparameters:")
    best_params = study.best_trial.params
    for k, v in sorted(best_params.items()):
        print(f"  {k:35s}: {v}")

    param_path = CFG["output_dir"] / "best_ann_params.json"
    with open(param_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    study.trials_dataframe().to_csv(
        CFG["output_dir"] / "ann_trial_log.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[5/6] Generating HPO diagnostic plots...")
    plot_hpo_results(study, CFG["output_dir"])

    # ── Retrain best ──────────────────────────────────────────────────────
    print("\n[6/6] Retraining best ANN at full budget...")
    result = retrain_best_ann(
        best_params, CFG, decoder, latent_dim,
        force_mat, inputs, force_scaler)
    plot_final_results(result, CFG, CFG["output_dir"])

    # ── Save full inference bundle ────────────────────────────────────────
    torch.save({
        "ann_state":       result["ann"].state_dict(),
        "decoder_state":   decoder.state_dict(),
        "cond_scaler":     result["cond_scaler"],
        "force_scaler":    force_scaler,
        "best_params":     best_params,
        "hidden_dims":     result["hidden_dims"],
        "angle_encoding":  result["angle_encoding"],
        "fourier_k":       result["fourier_k"],
        "latent_dim":      latent_dim,
        "metrics":         result["metrics"],
        "cfg":             CFG,
    }, CFG["output_dir"] / "best_ann_model.pt")
    print(f"  Inference bundle → {CFG['output_dir'] / 'best_ann_model.pt'}")

    print("\n" + "=" * 70)
    print(f"  HPO complete.  Best val MSE : {study.best_value:.6f}")
    m = result["metrics"]
    print(f"  Final test  →  R²={m['r2']:.5f}  RMSE={m['rmse']:.2f} N/m"
          f"  Rel.ε={m['rel_err']:.3f}%")
    print(f"  All outputs  : {CFG['output_dir'].resolve()}")
    print("=" * 70)
    return study, result


if __name__ == "__main__":
    study, result = main()


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE AFTER HPO
# ══════════════════════════════════════════════════════════════════════════════
"""
Load the optimised model and predict forces:

    import torch, numpy as np
    from hpo_ann_surrogate import FlexANN, FlexDecoder, encode_conditions

    ckpt   = torch.load("hpo_ann_results/best_ann_model.pt")
    bp     = ckpt["best_params"]
    latent = ckpt["latent_dim"]

    ann = FlexANN(in_dim=..., hidden_dims=ckpt["hidden_dims"],
                  out_dim=latent, act_name=bp["activation"],
                  norm_type=bp["norm_type"], dropout=0.0,
                  use_skip=bp["use_skip"])
    ann.load_state_dict(ckpt["ann_state"])
    ann.eval()

    # Single query
    inp      = np.array([[1000, 60, 90.0]])     # [RPM, Torque, angle_deg]
    enc, _, _ = encode_conditions(inp, ckpt["angle_encoding"],
                                   ckpt["fourier_k"], ckpt["cond_scaler"])
    with torch.no_grad():
        z_hat = ann(torch.tensor(enc))
        x_hat = decoder(z_hat).numpy()
    forces = ckpt["force_scaler"].inverse_transform(x_hat)[0]
    Fr = forces[:48];  Ft = forces[48:]

Resume HPO (add more trials without restarting):
    with open("hpo_ann_results/optuna_ann_study.pkl", "rb") as f:
        study = pickle.load(f)
    study.optimize(objective, n_trials=50)
"""
