"""
run_pod_prediction.py
=====================
End-to-end runner for the PODProcessor pipeline.

Steps
-----
1.  Train  – scan a folder of FEM xlsx files, build POD basis, train ANN.
2.  Save   – persist the trained model to disk.
3.  Predict – query any (rpm, torque) operating point.
4.  Evaluate – compare prediction against a known FEM result (optional).

Usage
-----
    python run_pod_prediction.py

Edit the USER CONFIG block below to match your paths and operating conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Import the processor class (adjust path if needed) ──────────────────────
from class_file_improved import PODProcessor


# ============================================================
# USER CONFIG  ← edit these
# ============================================================

# Folder containing your FEM xlsx files (e.g. "3000rpm_50Nm.xlsx")
DATA_DIR = Path("./fem_data")

# Where to save / load the trained model
MODEL_PATH = Path("./pod_ann_model.pt")

# Operating point to PREDICT (can be interpolated – does not need an xlsx)
PREDICT_RPM    = 3500      # rpm
PREDICT_TORQUE = 75.0      # Nm

# If you have an xlsx for the target OP and want to validate, set this.
# Otherwise set VALIDATION_XLSX = None to skip the FEM comparison.
VALIDATION_XLSX = None          # e.g. Path("./fem_data/3500rpm_75Nm.xlsx")
VALIDATION_TOOTH = 0            # 0-based tooth index to plot

# POD / ANN hyper-parameters
MODE_COUNT = 5      # number of POD modes retained
EPOCHS     = 800    # training epochs
LR         = 1e-3   # initial learning rate

# Set True to skip training and load an existing saved model instead
LOAD_EXISTING_MODEL = False


# ============================================================
# HELPER – pretty print a metrics dict
# ============================================================
def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20s}: {v:.6f}")
        elif isinstance(v, list):
            print(f"  {k:<20s}: [{v[0]:.6f} … {v[-1]:.6f}]  (len={len(v)})")
        else:
            print(f"  {k:<20s}: {v}")
    print()


# ============================================================
# STEP 1 – Initialise processor
# ============================================================
print("\n" + "="*60)
print("  POD-ANN Stator Force Surrogate")
print("="*60)

proc = PODProcessor(mode_count=MODE_COUNT, epochs=EPOCHS, lr=LR)


# ============================================================
# STEP 2 – Train or load
# ============================================================
if LOAD_EXISTING_MODEL and MODEL_PATH.exists():
    # ── Load previously trained model ───────────────────────
    print(f"\n[Load]  Loading model from '{MODEL_PATH}' …")
    proc.load(MODEL_PATH)
    print("[Load]  Done.")

else:
    # ── Run POD analysis on every xlsx in DATA_DIR ───────────
    print(f"\n[POD]  Scanning '{DATA_DIR}' for xlsx files …")
    pod_results = proc.run(DATA_DIR)

    print(f"\n[POD]  Collected {len(pod_results)} operating point(s):")
    for (rpm, torque), res in pod_results.items():
        cum_e = (res["s"][:MODE_COUNT] ** 2).sum() / (res["s"] ** 2).sum() * 100
        print(f"       {rpm:>5d} rpm  {torque:>8.1f} Nm  →  "
              f"{MODE_COUNT} modes capture {cum_e:.2f} % energy")

    # ── Train ANN ────────────────────────────────────────────
    print(f"\n[Train]  Training ANN for {EPOCHS} epochs …")
    history = proc.train_mode_predictor(test_size=0.2, random_state=42)
    print_metrics("ANN Test-Set Metrics", {
        k: v for k, v in history.items() if not isinstance(v, list)
    })

    # ── Plot training curves ─────────────────────────────────
    proc.plot_training_history(history)

    # ── Save model ───────────────────────────────────────────
    proc.save(MODEL_PATH)
    print(f"[Save]  Model saved to '{MODEL_PATH}'.")


# ============================================================
# STEP 3 – Predict for a target operating condition
# ============================================================
print(f"\n[Predict]  Operating point: {PREDICT_RPM} rpm, {PREDICT_TORQUE} Nm")

U_pred, T_pred = proc.predict_modes(PREDICT_RPM, PREDICT_TORQUE)
A_pred = proc.reconstruct_field(U_pred, T_pred)

print(f"[Predict]  Reconstructed force matrix shape: {A_pred.shape}")
print(f"           Rows = {A_pred.shape[0]} DOF  "
      f"(= {A_pred.shape[0]//2} teeth × 2 components FR+FT)")
print(f"           Cols = {A_pred.shape[1]} rotor-angle steps")


# ============================================================
# STEP 4 – Summary plot of all teeth for the predicted OP
# ============================================================
n_teeth = A_pred.shape[0] // 2
angle_steps = np.arange(A_pred.shape[1])

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for tooth in range(n_teeth):
    axes[0].plot(angle_steps, A_pred[tooth, :],
                 label=f"Tooth {tooth}", alpha=0.8)
    axes[1].plot(angle_steps, A_pred[n_teeth + tooth, :],
                 label=f"Tooth {tooth}", alpha=0.8)

axes[0].set_ylabel("Radial Force FR (normalised)")
axes[0].set_title(f"Predicted Radial Forces | {PREDICT_RPM} rpm | {PREDICT_TORQUE} Nm")
axes[0].legend(loc="upper right", fontsize=8, ncol=min(n_teeth, 4))
axes[0].grid(True)

axes[1].set_ylabel("Tangential Force FT (normalised)")
axes[1].set_title(f"Predicted Tangential Forces | {PREDICT_RPM} rpm | {PREDICT_TORQUE} Nm")
axes[1].set_xlabel("Rotor angle step")
axes[1].legend(loc="upper right", fontsize=8, ncol=min(n_teeth, 4))
axes[1].grid(True)

plt.suptitle("POD-ANN Prediction – All Teeth", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("predicted_forces_all_teeth.png", dpi=150)
plt.show()
print("[Plot]  All-teeth summary saved to 'predicted_forces_all_teeth.png'.")


# ============================================================
# STEP 5 – (Optional) Validate against FEM result
# ============================================================
if VALIDATION_XLSX is not None:
    print(f"\n[Validate]  Loading FEM reference: '{VALIDATION_XLSX}' …")

    df_val = proc.read_forces(VALIDATION_XLSX)
    A_actual = df_val.to_numpy(dtype=float).T   # (n_dof, n_steps)

    tooth_metrics = proc.evaluate_single_tooth(
        A_actual   = A_actual,
        A_pred     = A_pred,
        tooth_index= VALIDATION_TOOTH,
        rpm        = PREDICT_RPM,
        torque     = PREDICT_TORQUE,
        n_teeth    = n_teeth,
    )

    print_metrics(f"Tooth {VALIDATION_TOOTH} – Validation Metrics", tooth_metrics)

    # ── Per-tooth RMS across all teeth ───────────────────────
    print(f"\n[Validate]  Per-tooth RMS error (all {n_teeth} teeth):")
    print(f"  {'Tooth':>6}  {'FR RMS':>10}  {'FT RMS':>10}")
    for t in range(n_teeth):
        fr_rms = float(np.sqrt(
            ((A_actual[t, :] - A_pred[t, :]) ** 2).mean()
        ))
        ft_rms = float(np.sqrt(
            ((A_actual[n_teeth + t, :] - A_pred[n_teeth + t, :]) ** 2).mean()
        ))
        print(f"  {t:>6}  {fr_rms:>10.4f}  {ft_rms:>10.4f}")

else:
    print("\n[Validate]  No validation xlsx provided – skipping FEM comparison.")
    print("            Set VALIDATION_XLSX in the USER CONFIG block to enable it.")


print("\n" + "="*60)
print("  Done.")
print("="*60 + "\n")
