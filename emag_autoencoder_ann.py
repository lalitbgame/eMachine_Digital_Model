"""
================================================================================
  EV E-Machine Electromagnetic Force Prediction
  Architecture: Autoencoder (Encoder + Decoder) + ANN Surrogate

  Pipeline
  ────────
  Phase 1  ── Autoencoder training
               Raw forces x ∈ R^96  →  Encoder  →  Latent z ∈ R^d
               Latent z             →  Decoder  →  x̂ ≈ x   (MSE loss)

  Phase 2  ── ANN surrogate training (Decoder weights FROZEN)
               [RPM, Torque, θ_elec]  →  ANN  →  ẑ  →  Decoder  →  x̂
               Loss: MSE(x̂, x)  or  MSE(ẑ, z_enc)  (two strategies below)

  Inference── Encoder REMOVED entirely
               [RPM, Torque, θ_elec]  →  ANN  →  ẑ  →  Decoder  →  Fr[48], Ft[48]

  Machine specs
  ─────────────
  Stator teeth  : 48
  Force channels: Fr (radial) + Ft (tangential) per tooth  →  96-dim vector
  Speed         : 1000 RPM (fixed; extend trivially)
  Torque sweep  : configurable (e.g. 10 … 120 Nm)
  Angle range   : 0 … 360° electrical per cycle
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # Machine
    "n_teeth":          48,
    "n_force_comp":     2,           # [Fr, Ft]
    "input_dim":        96,          # 2 × 48
    "speed_rpm":        1000,

    # Dataset
    "torque_values":    [10, 20, 30, 40, 50, 60, 70, 80, 100, 120],
    "n_angles":         360,
    "data_path":        None,        # Path("your_data.csv") or None → synthetic

    # Autoencoder
    "latent_dim":       16,          # d — tune with reconstruction quality
    "ae_encoder_dims":  [96, 64, 32],  # hidden dims before bottleneck
    "ae_decoder_dims":  [32, 64, 96],  # hidden dims after bottleneck
    "ae_epochs":        400,
    "ae_lr":            1e-3,
    "ae_batch":         256,
    "ae_dropout":       0.10,
    "ae_weight_decay":  1e-5,

    # ANN surrogate
    "ann_input_dim":    3,           # [RPM, Torque, θ]
    "ann_hidden_dims":  [64, 128, 256, 128, 64],
    "ann_epochs":       500,
    "ann_lr":           1e-3,
    "ann_batch":        256,
    "ann_dropout":      0.10,
    "ann_weight_decay": 1e-5,
    # 'latent'    → ANN trained with MSE(ẑ, z_enc)  [fast, decoupled]
    # 'end_to_end'→ ANN+Decoder jointly with MSE(x̂, x)  [slightly better]
    "ann_loss_mode":    "end_to_end",

    # Training
    "train_split":      0.75,
    "val_split":        0.15,
    "random_seed":      42,
    "output_dir":       Path("ae_ann_results"),
}

torch.manual_seed(CFG["random_seed"])
np.random.seed(CFG["random_seed"])
CFG["output_dir"].mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA GENERATOR  (replace with your FEA / measured CSV)
# ══════════════════════════════════════════════════════════════════════════════
def load_or_generate_data(cfg: dict) -> dict:
    """
    Returns
    -------
    force_mat : np.ndarray  [N_snap × 96]   normalised force vectors
    inputs    : np.ndarray  [N_snap × 3]    [rpm, torque, angle_deg]
    scaler    : StandardScaler fitted on force_mat  (for un-scaling later)

    CSV format expected (one row = one snapshot):
        speed_rpm, torque_nm, angle_deg,
        Fr_t1 … Fr_t48,
        Ft_t1 … Ft_t48
    """
    if cfg["data_path"] and Path(cfg["data_path"]).exists():
        df     = pd.read_csv(cfg["data_path"])
        fr_cols = [f"Fr_t{i+1}" for i in range(cfg["n_teeth"])]
        ft_cols = [f"Ft_t{i+1}" for i in range(cfg["n_teeth"])]
        Fr     = df[fr_cols].values                    # [N × 48]
        Ft     = df[ft_cols].values                    # [N × 48]
        inputs = df[["speed_rpm","torque_nm","angle_deg"]].values
    else:
        print("  [INFO] No data path — generating synthetic eMag force data.")
        Fr, Ft, inputs = _synthetic_emag(cfg)

    force_mat = np.hstack([Fr, Ft]).astype(np.float32)  # [N × 96]
    scaler    = StandardScaler()
    force_mat = scaler.fit_transform(force_mat)
    print(f"  Snapshots: {force_mat.shape[0]}   Force dim: {force_mat.shape[1]}")
    return {"force_mat": force_mat, "inputs": inputs.astype(np.float32),
            "scaler": scaler, "Fr_raw": Fr, "Ft_raw": Ft}


def _synthetic_emag(cfg):
    """
    Physics-inspired synthetic eMag forces for a 48-slot IPMSM.
    Dominant harmonic orders: 1, 6, 12, 24 (tooth-passing harmonics).
    Torque dependency: amplitude scales linearly with load.
    """
    n   = cfg["n_teeth"]
    ang = np.linspace(0, 360, cfg["n_angles"], endpoint=False)
    ti  = np.arange(n)
    sp  = 2 * np.pi * ti / n        # spatial phase per tooth

    Fr_list, Ft_list, inp_list = [], [], []
    rng = np.random.default_rng(cfg["random_seed"])

    for T in cfg["torque_values"]:
        for a_deg in ang:
            ar = np.deg2rad(a_deg)
            Fr = (
                (800 + 5.0*T)
                + (200 + 1.8*T) * np.cos(  sp + ar)
                + ( 80 + 0.7*T) * np.cos(6*sp + 6*ar)
                + ( 35 + 0.3*T) * np.cos(12*sp + 12*ar)
                + ( 15 + 0.1*T) * np.cos(24*sp + 24*ar)
                + rng.normal(0, 4, n)
            )
            Ft = (
                  (0.9*T) * np.sin(  sp + ar)
                + (0.4*T) * np.sin(6*sp + 6*ar)
                + (0.15*T)* np.sin(12*sp + 12*ar)
                + rng.normal(0, 2, n)
            )
            Fr_list.append(Fr)
            Ft_list.append(Ft)
            inp_list.append([cfg["speed_rpm"], T, a_deg])

    return (np.array(Fr_list, dtype=np.float32),
            np.array(Ft_list, dtype=np.float32),
            np.array(inp_list, dtype=np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# 2. AUTOENCODER  (Encoder + Decoder as independent nn.Module objects)
# ══════════════════════════════════════════════════════════════════════════════
def _mlp_block(in_d: int, out_d: int, dropout: float, act=nn.GELU) -> nn.Sequential:
    """Linear → BatchNorm → Activation → Dropout"""
    return nn.Sequential(
        nn.Linear(in_d, out_d),
        nn.BatchNorm1d(out_d),
        act(),
        nn.Dropout(dropout),
    )


class Encoder(nn.Module):
    """
    Maps x ∈ R^input_dim  →  z ∈ R^latent_dim.

    Architecture: stacked MLP blocks with progressively shrinking width.
    Final layer is a plain Linear (no activation) so the latent space
    is unconstrained — the decoder learns to interpret the geometry.
    """
    def __init__(self, input_dim: int, hidden_dims: list,
                 latent_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev   = input_dim
        for h in hidden_dims:
            layers.append(_mlp_block(prev, h, dropout))
            prev = h
        layers.append(nn.Linear(prev, latent_dim))   # bottleneck (no BN/act)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Maps z ∈ R^latent_dim  →  x̂ ∈ R^input_dim.

    Mirror of encoder. Output is linear (no sigmoid/tanh) because
    forces are standardised — no hard bounds.
    """
    def __init__(self, latent_dim: int, hidden_dims: list,
                 output_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev   = latent_dim
        for h in hidden_dims:
            layers.append(_mlp_block(prev, h, dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANN SURROGATE
# ══════════════════════════════════════════════════════════════════════════════
class ANNSurrogate(nn.Module):
    """
    Maps operational conditions [RPM, Torque, θ_elec]  →  latent ẑ ∈ R^d.

    Uses a residual skip connection from input to output to handle the
    near-linear torque–amplitude relationship cleanly.
    GELU activation is smooth and works well for physics surrogates.
    """
    def __init__(self, in_dim: int, hidden_dims: list,
                 out_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev   = in_dim
        for h in hidden_dims:
            layers.append(_mlp_block(prev, h, dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net  = nn.Sequential(*layers)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)  # residual path

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond) + self.skip(cond)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASET HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def make_splits(force_mat, inputs, cfg):
    n     = len(force_mat)
    n_tr  = int(n * cfg["train_split"])
    n_va  = int(n * cfg["val_split"])
    n_te  = n - n_tr - n_va

    force_t  = torch.tensor(force_mat)
    inputs_t = torch.tensor(inputs)

    ds  = TensorDataset(force_t, inputs_t)
    g   = torch.Generator().manual_seed(cfg["random_seed"])
    tr, va, te = random_split(ds, [n_tr, n_va, n_te], generator=g)
    return tr, va, te


def loader(ds, batch_size, shuffle=True):
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, pin_memory=True, drop_last=False)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING ROUTINES
# ══════════════════════════════════════════════════════════════════════════════

# ── 5a. Phase 1: Autoencoder ─────────────────────────────────────────────────
def train_autoencoder(cfg, train_ds, val_ds) -> Autoencoder:
    encoder = Encoder(cfg["input_dim"], cfg["ae_encoder_dims"],
                      cfg["latent_dim"], cfg["ae_dropout"])
    decoder = Decoder(cfg["latent_dim"], cfg["ae_decoder_dims"],
                      cfg["input_dim"], cfg["ae_dropout"])
    ae      = Autoencoder(encoder, decoder).to(DEVICE)

    opt   = optim.AdamW(ae.parameters(), lr=cfg["ae_lr"],
                        weight_decay=cfg["ae_weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["ae_epochs"],
                                                  eta_min=cfg["ae_lr"] * 0.01)
    crit  = nn.MSELoss()

    tr_ld = loader(train_ds, cfg["ae_batch"])
    va_ld = loader(val_ds,   512, shuffle=False)
    history = {"train": [], "val": []}
    best_val, best_sd = np.inf, None

    print("\n── Phase 1: Autoencoder training ──────────────────────────────")
    for epoch in range(1, cfg["ae_epochs"] + 1):
        ae.train()
        tr_loss = 0.0
        for x_batch, _ in tr_ld:
            x_batch = x_batch.to(DEVICE)
            opt.zero_grad()
            x_hat, _ = ae(x_batch)
            loss = crit(x_hat, x_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        ae.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x_batch, _ in va_ld:
                x_hat, _ = ae(x_batch.to(DEVICE))
                va_loss += crit(x_hat, x_batch.to(DEVICE)).item()
        va_loss /= len(va_ld)

        sched.step()
        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_sd  = {k: v.clone() for k, v in ae.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{cfg['ae_epochs']}  "
                  f"train: {tr_loss:.6f}  val: {va_loss:.6f}")

    ae.load_state_dict(best_sd)
    print(f"  Best AE val loss: {best_val:.6f}")
    return ae, history


# ── 5b. Collect latent codes from trained encoder ────────────────────────────
@torch.no_grad()
def extract_latent_codes(encoder: Encoder, dataset) -> tuple:
    """
    Run the full dataset through the frozen encoder.
    Returns (latent_codes [N × d], force_vecs [N × 96], op_conds [N × 3]).
    """
    encoder.eval()
    ld = loader(dataset, 512, shuffle=False)
    zs, xs, cs = [], [], []
    for x_batch, c_batch in ld:
        zs.append(encoder(x_batch.to(DEVICE)).cpu())
        xs.append(x_batch)
        cs.append(c_batch)
    return (torch.cat(zs).numpy(),
            torch.cat(xs).numpy(),
            torch.cat(cs).numpy())


# ── 5c. Phase 2: ANN surrogate ───────────────────────────────────────────────
def train_ann_surrogate(cfg, ae: Autoencoder,
                        train_ds, val_ds) -> ANNSurrogate:
    """
    Two modes controlled by cfg["ann_loss_mode"]:

    'latent'      : freeze encoder+decoder, extract z codes once,
                    train ANN with MSELoss(ẑ, z_enc).
                    Fast. Decoupled. Works when AE reconstruction is excellent.

    'end_to_end'  : freeze decoder only, train ANN with
                    MSELoss(Decoder(ẑ), x).
                    Slightly more accurate — ANN sees reconstruction quality directly.
    """
    # Build ANN
    # Normalise condition inputs
    cond_scaler = StandardScaler()
    all_conds   = torch.cat([c for _, c in loader(train_ds, len(train_ds),
                                                    shuffle=False)]).numpy()
    cond_scaler.fit(all_conds)

    ann = ANNSurrogate(cfg["ann_input_dim"], cfg["ann_hidden_dims"],
                       cfg["latent_dim"], cfg["ann_dropout"]).to(DEVICE)

    # Freeze encoder always; freeze decoder for both modes
    for p in ae.encoder.parameters():
        p.requires_grad = False
    for p in ae.decoder.parameters():
        p.requires_grad = False

    opt   = optim.AdamW(ann.parameters(), lr=cfg["ann_lr"],
                        weight_decay=cfg["ann_weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["ann_epochs"],
                                                   eta_min=cfg["ann_lr"] * 0.01)
    crit  = nn.MSELoss()

    print(f"\n── Phase 2: ANN surrogate training  (mode: {cfg['ann_loss_mode']}) ──")

    if cfg["ann_loss_mode"] == "latent":
        # Pre-extract latent codes — only done once
        z_train, _, c_train = extract_latent_codes(ae.encoder, train_ds)
        z_val,   _, c_val   = extract_latent_codes(ae.encoder, val_ds)

        c_train_sc = cond_scaler.transform(c_train).astype(np.float32)
        c_val_sc   = cond_scaler.transform(c_val).astype(np.float32)

        tr_ld2 = DataLoader(
            TensorDataset(torch.tensor(c_train_sc), torch.tensor(z_train)),
            batch_size=cfg["ann_batch"], shuffle=True)
        va_ld2 = DataLoader(
            TensorDataset(torch.tensor(c_val_sc), torch.tensor(z_val)),
            batch_size=512)

        history = {"train": [], "val": []}
        best_val, best_sd = np.inf, None

        for epoch in range(1, cfg["ann_epochs"] + 1):
            ann.train()
            tr_loss = sum(
                _step_latent(ann, opt, crit, cb, zb)
                for cb, zb in tr_ld2
            ) / len(tr_ld2)

            ann.eval()
            with torch.no_grad():
                va_loss = sum(
                    crit(ann(cb.to(DEVICE)), zb.to(DEVICE)).item()
                    for cb, zb in va_ld2
                ) / len(va_ld2)

            sched.step()
            history["train"].append(tr_loss)
            history["val"].append(va_loss)
            if va_loss < best_val:
                best_val = va_loss
                best_sd  = {k: v.clone() for k, v in ann.state_dict().items()}
            if epoch % 50 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{cfg['ann_epochs']}  "
                      f"train: {tr_loss:.6f}  val: {va_loss:.6f}")

    else:  # end_to_end
        tr_ld2 = loader(train_ds, cfg["ann_batch"])
        va_ld2 = loader(val_ds,   512, shuffle=False)

        history = {"train": [], "val": []}
        best_val, best_sd = np.inf, None

        for epoch in range(1, cfg["ann_epochs"] + 1):
            ann.train()
            tr_loss = 0.0
            for x_batch, c_batch in tr_ld2:
                x_batch = x_batch.to(DEVICE)
                c_batch = c_batch.to(DEVICE)
                c_sc    = _scale_cond(cond_scaler, c_batch)
                opt.zero_grad()
                z_hat   = ann(c_sc)
                x_hat   = ae.decoder(z_hat)
                loss    = crit(x_hat, x_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(ann.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item()
            tr_loss /= len(tr_ld2)

            ann.eval()
            va_loss = 0.0
            with torch.no_grad():
                for x_batch, c_batch in va_ld2:
                    x_batch = x_batch.to(DEVICE)
                    c_sc    = _scale_cond(cond_scaler, c_batch.to(DEVICE))
                    z_hat   = ann(c_sc)
                    x_hat   = ae.decoder(z_hat)
                    va_loss += crit(x_hat, x_batch).item()
            va_loss /= len(va_ld2)

            sched.step()
            history["train"].append(tr_loss)
            history["val"].append(va_loss)
            if va_loss < best_val:
                best_val = va_loss
                best_sd  = {k: v.clone() for k, v in ann.state_dict().items()}
            if epoch % 50 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{cfg['ann_epochs']}  "
                      f"train: {tr_loss:.6f}  val: {va_loss:.6f}")

    ann.load_state_dict(best_sd)
    print(f"  Best ANN val loss: {best_val:.6f}")
    return ann, cond_scaler, history


def _step_latent(ann, opt, crit, cb, zb):
    cb, zb = cb.to(DEVICE), zb.to(DEVICE)
    opt.zero_grad()
    loss = crit(ann(cb), zb)
    loss.backward()
    nn.utils.clip_grad_norm_(ann.parameters(), 1.0)
    opt.step()
    return loss.item()


def _scale_cond(scaler: StandardScaler, c: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=c.device)
    std  = torch.tensor(scaler.scale_, dtype=torch.float32, device=c.device)
    return (c - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_full_pipeline(ann: ANNSurrogate, decoder: Decoder,
                           cond_scaler: StandardScaler,
                           force_scaler: StandardScaler,
                           test_ds, cfg) -> dict:
    """
    ANN → ẑ → Decoder → x̂  evaluated on test split.
    Reports R², RMSE, relative error in PHYSICAL (un-scaled) units.
    """
    ann.eval();  decoder.eval()
    te_ld = loader(test_ds, 512, shuffle=False)

    x_true_list, x_pred_list = [], []
    for x_batch, c_batch in te_ld:
        c_sc  = _scale_cond(cond_scaler, c_batch.to(DEVICE))
        z_hat = ann(c_sc)
        x_hat = decoder(z_hat).cpu().numpy()
        x_true_list.append(x_batch.numpy())
        x_pred_list.append(x_hat)

    x_true_sc = np.vstack(x_true_list)
    x_pred_sc = np.vstack(x_pred_list)

    # Un-scale to physical units
    x_true = force_scaler.inverse_transform(x_true_sc)
    x_pred = force_scaler.inverse_transform(x_pred_sc)

    r2   = r2_score(x_true.ravel(), x_pred.ravel())
    rmse = np.sqrt(mean_squared_error(x_true.ravel(), x_pred.ravel()))
    rel  = (np.linalg.norm(x_true - x_pred, "fro") /
            np.linalg.norm(x_true, "fro") * 100)

    n = cfg["n_teeth"]
    r2_fr = r2_score(x_true[:, :n].ravel(), x_pred[:, :n].ravel())
    r2_ft = r2_score(x_true[:, n:].ravel(), x_pred[:, n:].ravel())

    print("\n  ── Test-set evaluation (physical units) ──────────────────")
    print(f"     Overall R²    : {r2:.6f}")
    print(f"     R² (Fr only)  : {r2_fr:.6f}")
    print(f"     R² (Ft only)  : {r2_ft:.6f}")
    print(f"     RMSE          : {rmse:.4f}  N/m")
    print(f"     Rel. error ε  : {rel:.4f} %")

    return {"r2": r2, "r2_fr": r2_fr, "r2_ft": r2_ft,
            "rmse": rmse, "rel_err": rel,
            "x_true": x_true, "x_pred": x_pred}


# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(ae_hist, ann_hist, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Training histories", fontsize=13, fontweight="bold")

    for ax, hist, title in zip(
            axes,
            [ae_hist, ann_hist],
            ["Phase 1 — Autoencoder", "Phase 2 — ANN surrogate"]):
        ep = np.arange(1, len(hist["train"]) + 1)
        ax.semilogy(ep, hist["train"], label="Train", color="#1D9E75")
        ax.semilogy(ep, hist["val"],   label="Val",   color="#D85A30")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (log)")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_dir / 'training_curves.png'}")


def plot_latent_space(z_all, inputs_all, cfg, out_dir):
    """
    2-D scatter of the first two latent dimensions, coloured by torque and angle.
    Gives a quick sanity-check that the encoder has learned meaningful structure.
    """
    if z_all.shape[1] < 2:
        return
    torques = inputs_all[:, 1]
    angles  = inputs_all[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Latent space structure (z₁ vs z₂)", fontsize=13, fontweight="bold")

    sc = axes[0].scatter(z_all[:, 0], z_all[:, 1], c=torques,
                         cmap="viridis", s=2, alpha=0.5)
    plt.colorbar(sc, ax=axes[0], label="Torque [Nm]")
    axes[0].set_xlabel("z₁"); axes[0].set_ylabel("z₂")
    axes[0].set_title("Coloured by torque"); axes[0].grid(alpha=0.3)

    sc = axes[1].scatter(z_all[:, 0], z_all[:, 1], c=angles,
                         cmap="hsv", s=2, alpha=0.5)
    plt.colorbar(sc, ax=axes[1], label="Electrical angle [deg]")
    axes[1].set_xlabel("z₁"); axes[1].set_ylabel("z₂")
    axes[1].set_title("Coloured by angle"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "latent_space.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_dir / 'latent_space.png'}")


def plot_reconstruction_vs_prediction(ae: Autoencoder, eval_res: dict,
                                       force_scaler, cfg, out_dir, snap=0):
    """
    Three-way comparison for a single test snapshot:
       FEA ground truth  |  AE reconstruction  |  ANN+Decoder prediction
    """
    n = cfg["n_teeth"]
    x_true = eval_res["x_true"]
    x_pred = eval_res["x_pred"]

    # AE reconstruction of that snapshot (using encoder+decoder)
    x_sc    = force_scaler.transform(x_true[[snap]])
    with torch.no_grad():
        x_hat_ae, _ = ae(torch.tensor(x_sc, dtype=torch.float32).to(DEVICE))
    x_ae = force_scaler.inverse_transform(x_hat_ae.cpu().numpy())

    tooth_idx = np.arange(n)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Ground truth vs AE recon vs ANN prediction — snapshot {snap}\n"
        f"Overall R² = {eval_res['r2']:.5f}",
        fontsize=13, fontweight="bold")

    def _plot_row(ax_left, ax_right, key, sl, label_y):
        ax_left.plot(tooth_idx, x_true[snap, sl], "o-", ms=3,
                     label="FEA truth",    color="#185FA5", lw=1.5)
        ax_left.plot(tooth_idx, x_ae[0, sl], "s--", ms=3,
                     label="AE recon",     color="#1D9E75", lw=1.4)
        ax_left.plot(tooth_idx, x_pred[snap, sl], "^:", ms=3,
                     label="ANN predict",  color="#D85A30", lw=1.4)
        ax_left.set_xlabel("Tooth index"); ax_left.set_ylabel(label_y)
        ax_left.set_title(f"{label_y} per tooth"); ax_left.legend(fontsize=9)
        ax_left.grid(alpha=0.3)

        lim = [x_true[:, sl].min(), x_true[:, sl].max()]
        ax_right.scatter(x_true[:, sl].ravel(), x_pred[:, sl].ravel(),
                         s=1, alpha=0.3, color="#D85A30")
        ax_right.plot(lim, lim, "k--", lw=1)
        ax_right.set_xlabel(f"FEA {label_y}"); ax_right.set_ylabel(f"Predicted {label_y}")
        ax_right.set_title(f"Parity — {label_y}"); ax_right.grid(alpha=0.3)

    _plot_row(axes[0, 0], axes[0, 1], "Fr", slice(None, n), "Fr [N/m]")
    _plot_row(axes[1, 0], axes[1, 1], "Ft", slice(n, None), "Ft [N/m]")

    plt.tight_layout()
    fig.savefig(out_dir / "force_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_dir / 'force_comparison.png'}")


def plot_polar_forces(eval_res: dict, cfg: dict, out_dir, snap=0):
    n     = cfg["n_teeth"]
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    theta = np.append(theta, theta[0])

    x_true = eval_res["x_true"]
    x_pred = eval_res["x_pred"]

    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(12, 5))
    fig.suptitle(f"Polar tooth-force distribution — snapshot {snap}", fontsize=12)

    for ax, sl, lbl in zip(axes, [slice(None, n), slice(n, None)], ["Fr", "Ft"]):
        t = np.append(x_true[snap, sl], x_true[snap, sl][0])
        p = np.append(x_pred[snap, sl], x_pred[snap, sl][0])
        ax.plot(theta, t, lw=1.8, label="FEA", color="#185FA5")
        ax.plot(theta, p, lw=1.4, ls="--", label="ANN-Dec", color="#D85A30")
        ax.set_title(lbl, pad=14); ax.legend(loc="upper right", fontsize=8)
        ax.set_theta_zero_location("N")

    plt.tight_layout()
    fig.savefig(out_dir / "polar_forces.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_dir / 'polar_forces.png'}")


def plot_error_heatmap(eval_res: dict, cfg: dict, inputs_test, out_dir):
    """
    Heatmap of per-tooth relative error across all test snapshots,
    averaged by torque level — useful for spotting systematic bias.
    """
    n       = cfg["n_teeth"]
    x_true  = eval_res["x_true"]
    x_pred  = eval_res["x_pred"]
    torques = inputs_test[:, 1]
    unique_T = np.unique(torques)

    err_grid = np.zeros((len(unique_T), n))
    for i, T in enumerate(unique_T):
        mask = torques == T
        if mask.sum() == 0:
            continue
        err_grid[i] = (
            np.abs(x_true[mask, :n] - x_pred[mask, :n]).mean(axis=0)
            / (np.abs(x_true[mask, :n]).mean(axis=0) + 1e-8) * 100
        )

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(err_grid, aspect="auto", cmap="Reds",
                   origin="lower", vmin=0)
    plt.colorbar(im, ax=ax, label="Mean relative error Fr [%]")
    ax.set_xticks(np.arange(0, n, 4))
    ax.set_xticklabels(np.arange(1, n+1, 4), fontsize=8)
    ax.set_yticks(np.arange(len(unique_T)))
    ax.set_yticklabels([f"{int(T)} Nm" for T in unique_T], fontsize=9)
    ax.set_xlabel("Tooth index"); ax.set_ylabel("Torque level")
    ax.set_title("Per-tooth relative error in Fr by torque (test set)")
    plt.tight_layout()
    fig.savefig(out_dir / "error_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_dir / 'error_heatmap.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. INFERENCE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
class EmagPredictor:
    """
    Clean inference interface — encoder completely absent.

    Usage
    -----
    pred = EmagPredictor(ann, decoder, cond_scaler, force_scaler, cfg)
    Fr, Ft = pred.predict(torque_nm=60, angle_deg=90.0)
    result = pred.cycle(torque_nm=60, n_steps=720)
    """
    def __init__(self, ann: ANNSurrogate, decoder: Decoder,
                 cond_scaler: StandardScaler, force_scaler: StandardScaler,
                 cfg: dict):
        self.ann          = ann.eval()
        self.decoder      = decoder.eval()
        self.cond_scaler  = cond_scaler
        self.force_scaler = force_scaler
        self.cfg          = cfg

    @torch.no_grad()
    def predict(self, torque_nm: float, angle_deg: float,
                speed_rpm: float = None) -> tuple:
        """Single operating point → Fr[48], Ft[48] in N/m."""
        rpm = speed_rpm or self.cfg["speed_rpm"]
        c   = np.array([[rpm, torque_nm, angle_deg]], dtype=np.float32)
        c_sc = self.cond_scaler.transform(c)
        c_t  = torch.tensor(c_sc).to(DEVICE)
        z_hat = self.ann(c_t)
        x_hat = self.decoder(z_hat).cpu().numpy()
        x_phy = self.force_scaler.inverse_transform(x_hat)[0]
        n = self.cfg["n_teeth"]
        return x_phy[:n], x_phy[n:]

    @torch.no_grad()
    def cycle(self, torque_nm: float, n_steps: int = 360,
              speed_rpm: float = None) -> dict:
        """
        Full electrical cycle prediction for all 48 teeth.
        Returns { 'angles': [n_steps], 'Fr': [48 × n_steps], 'Ft': [48 × n_steps] }
        """
        rpm    = speed_rpm or self.cfg["speed_rpm"]
        angles = np.linspace(0, 360, n_steps, endpoint=False)
        c      = np.column_stack([
                     np.full(n_steps, rpm),
                     np.full(n_steps, torque_nm),
                     angles]).astype(np.float32)
        c_sc   = self.cond_scaler.transform(c)
        c_t    = torch.tensor(c_sc).to(DEVICE)
        z_hat  = self.ann(c_t)
        x_hat  = self.decoder(z_hat).cpu().numpy()
        x_phy  = self.force_scaler.inverse_transform(x_hat)
        n      = self.cfg["n_teeth"]
        return {"angles": angles, "Fr": x_phy[:, :n].T, "Ft": x_phy[:, n:].T}


# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════
def save_all(ae, ann, cond_scaler, force_scaler, cfg, out_dir):
    torch.save({
        "encoder_state":  ae.encoder.state_dict(),
        "decoder_state":  ae.decoder.state_dict(),
        "ann_state":      ann.state_dict(),
        "cond_scaler":    cond_scaler,
        "force_scaler":   force_scaler,
        "cfg":            cfg,
    }, out_dir / "emag_ae_ann_model.pt")
    print(f"  Model bundle saved → {out_dir / 'emag_ae_ann_model.pt'}")


def load_predictor(model_path: Path) -> EmagPredictor:
    """Reload a saved model and return an EmagPredictor ready for inference."""
    ckpt = torch.load(model_path, map_location=DEVICE)
    cfg  = ckpt["cfg"]
    enc  = Encoder(cfg["input_dim"], cfg["ae_encoder_dims"],
                   cfg["latent_dim"], cfg["ae_dropout"])
    dec  = Decoder(cfg["latent_dim"], cfg["ae_decoder_dims"],
                   cfg["input_dim"], cfg["ae_dropout"])
    ann  = ANNSurrogate(cfg["ann_input_dim"], cfg["ann_hidden_dims"],
                        cfg["latent_dim"], cfg["ann_dropout"])
    enc.load_state_dict(ckpt["encoder_state"])
    dec.load_state_dict(ckpt["decoder_state"])
    ann.load_state_dict(ckpt["ann_state"])
    dec.to(DEVICE);  ann.to(DEVICE)
    return EmagPredictor(ann, dec, ckpt["cond_scaler"],
                         ckpt["force_scaler"], cfg)


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  EV E-Machine eMag Forces: Autoencoder + ANN Surrogate Pipeline")
    print("=" * 70)

    # ── Load / generate data ──────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    data = load_or_generate_data(CFG)
    force_mat = data["force_mat"]
    inputs    = data["inputs"]

    train_ds, val_ds, test_ds = make_splits(force_mat, inputs, CFG)

    # ── Phase 1: Autoencoder ─────────────────────────────────────────────
    print("\n[2/6] Phase 1 — Autoencoder")
    ae, ae_hist = train_autoencoder(CFG, train_ds, val_ds)

    # AE reconstruction quality on test set
    ae.eval()
    te_ld = loader(test_ds, 512, shuffle=False)
    ae_preds, ae_trues = [], []
    with torch.no_grad():
        for x_b, _ in te_ld:
            x_hat, _ = ae(x_b.to(DEVICE))
            ae_preds.append(x_hat.cpu().numpy())
            ae_trues.append(x_b.numpy())
    ae_preds = np.vstack(ae_preds)
    ae_trues = np.vstack(ae_trues)
    ae_r2 = r2_score(ae_trues.ravel(), ae_preds.ravel())
    print(f"\n  AE test reconstruction  R² = {ae_r2:.6f}")

    # ── Extract latent codes (for visualisation) ─────────────────────────
    print("\n[3/6] Extracting latent codes from full dataset...")
    full_ds = TensorDataset(torch.tensor(force_mat), torch.tensor(inputs))
    z_all, _, inp_all = extract_latent_codes(ae.encoder, full_ds)
    plot_latent_space(z_all, inp_all, CFG, CFG["output_dir"])

    # ── Phase 2: ANN surrogate ────────────────────────────────────────────
    print("\n[4/6] Phase 2 — ANN surrogate")
    ann, cond_scaler, ann_hist = train_ann_surrogate(CFG, ae, train_ds, val_ds)

    # ── Training curves ───────────────────────────────────────────────────
    plot_training_curves(ae_hist, ann_hist, CFG["output_dir"])

    # ── Full pipeline evaluation ──────────────────────────────────────────
    print("\n[5/6] Evaluating ANN + Decoder pipeline on test set...")
    te_inputs = torch.cat([c for _, c in loader(test_ds, len(test_ds),
                                                  shuffle=False)]).numpy()
    eval_res = evaluate_full_pipeline(
        ann, ae.decoder, cond_scaler, data["scaler"], test_ds, CFG)

    plot_reconstruction_vs_prediction(ae, eval_res, data["scaler"], CFG,
                                       CFG["output_dir"])
    plot_polar_forces(eval_res, CFG, CFG["output_dir"])
    plot_error_heatmap(eval_res, CFG, te_inputs, CFG["output_dir"])

    # ── Save model ────────────────────────────────────────────────────────
    print("\n[6/6] Saving model...")
    save_all(ae, ann, cond_scaler, data["scaler"], CFG, CFG["output_dir"])

    # ── Demo inference (encoder completely removed) ───────────────────────
    predictor = EmagPredictor(ann, ae.decoder, cond_scaler,
                               data["scaler"], CFG)

    Fr_demo, Ft_demo = predictor.predict(torque_nm=60, angle_deg=90.0)
    print(f"\n  Demo  (T=60 Nm, θ=90°):")
    print(f"    Fr  max={Fr_demo.max():.1f}  min={Fr_demo.min():.1f}  N/m")
    print(f"    Ft  max={Ft_demo.max():.1f}  min={Ft_demo.min():.1f}  N/m")

    cycle = predictor.cycle(torque_nm=60, n_steps=720)
    print(f"\n  Cycle prediction shapes:  Fr {cycle['Fr'].shape}  Ft {cycle['Ft'].shape}")
    print(f"  (48 teeth × 720 angle steps — microseconds per query)")

    print("\n" + "=" * 70)
    print(f"  Done.  Results in: {CFG['output_dir'].resolve()}")
    print("=" * 70)
    return predictor


if __name__ == "__main__":
    predictor = main()


# ══════════════════════════════════════════════════════════════════════════════
# QUICK-START GUIDE
# ══════════════════════════════════════════════════════════════════════════════
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PLUG IN YOUR DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CSV (one row = one operating snapshot):
  speed_rpm, torque_nm, angle_deg, Fr_t1, ..., Fr_t48, Ft_t1, ..., Ft_t48

Then set:
  CFG["data_path"] = Path("my_fea_data.csv")

Multiple torque files (common FEA export pattern):
  frames = []
  for T in torque_list:
      df = pd.read_csv(f"T{T}Nm_1000rpm.csv")
      df["torque_nm"] = T;  df["speed_rpm"] = 1000
      frames.append(df)
  pd.concat(frames).to_csv("combined.csv", index=False)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. KEY HYPERPARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  latent_dim        Start at 16. Reduce if latent-space plot shows clean
                    cluster structure. Increase if AE R² < 0.999.

  ae_encoder_dims   [96,64,32] is a good default. For noisy data add a
                    wider first layer: [96,128,64,32].

  ann_loss_mode     'end_to_end' gives ~0.2–0.5% lower final error.
                    'latent' converges 2× faster — useful for prototyping.

  ann_hidden_dims   [64,128,256,128,64] for strong nonlinearity.
                    Reduce to [64,128,64] if dataset < 5000 snapshots.

  ae_dropout /      Start at 0.10. Increase to 0.20 if you see
  ann_dropout       overfitting on small datasets.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. INFERENCE (encoder removed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  pred = load_predictor(Path("ae_ann_results/emag_ae_ann_model.pt"))

  # Single point
  Fr, Ft = pred.predict(torque_nm=80, angle_deg=45.0)

  # Full electrical cycle
  result = pred.cycle(torque_nm=80, n_steps=720)
  # result["Fr"] → shape [48, 720]  N/m
  # result["Ft"] → shape [48, 720]  N/m
  # result["angles"] → shape [720]  degrees

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. EXTENDING TO MULTIPLE SPEEDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ANN input is already [RPM, Torque, θ] — just include speed in your
  training snapshots and set ann_hidden_dims slightly wider:
    CFG["ann_hidden_dims"] = [64, 128, 256, 256, 128, 64]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
