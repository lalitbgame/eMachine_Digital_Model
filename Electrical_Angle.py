import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import svd
from scipy.fft import rfft, rfftfreq

# ==========================================
# Configuration & Physics Constants
# ==========================================
DATA_FOLDER = './emag_data'
COMMON_GRID_POINTS = 3601    # 0.1 degree resolution
ANGLE_RANGE = (0, 360)       # Electrical angle 0 to 360
POD_MODES_TO_KEEP = 5        # Number of modes to test for coverage

class EmagForceAnalyzer:
    """
    Handles parsing, visualization, POD, and Spectral Coverage analysis.
    """
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.common_angle = np.linspace(ANGLE_RANGE[0], ANGLE_RANGE[1], COMMON_GRID_POINTS)
        self.snapshot_matrix = None 
        self.operating_conditions = [] 
        self.U = None
        self.S = None
        self.Vt = None

    def _parse_filename(self, filename):
        rpm_match = re.search(r"(\d+\.?\d*)\s*[_]?rpm", filename, re.IGNORECASE)
        trq_match = re.search(r"(\d+\.?\d*)\s*[_]?nm", filename, re.IGNORECASE)
        if rpm_match and trq_match:
            return float(rpm_match.group(1)), float(trq_match.group(1))
        return None, None

    def load_and_process_data(self):
        search_path = os.path.join(self.folder_path, "*.xlsx")
        files = glob.glob(search_path)
        
        if not files:
            print(f"No .xlsx files found in {self.folder_path}")
            return

        snapshots = []
        
        for filepath in files:
            fname = os.path.basename(filepath)
            rpm, torque = self._parse_filename(fname)
            
            if rpm is None: continue
            
            try:
                df = pd.read_excel(filepath)
                # Resample Force on Tooth 1 (Representative)
                # For simplicity in this demo, we focus on Tooth 1 spectral coverage
                f_interp = interp1d(df['Electrical_Angle'], df['Force_Tooth_1'], 
                                    kind='cubic', fill_value="extrapolate")
                force_clean = f_interp(self.common_angle)
                
                snapshots.append(force_clean)
                self.operating_conditions.append((rpm, torque))
                
            except Exception as e:
                print(f"Error processing {fname}: {e}")

        if snapshots:
            self.snapshot_matrix = np.array(snapshots).T 
            print(f"Data Loaded. Snapshot Matrix Shape: {self.snapshot_matrix.shape}")

    def perform_pod_analysis(self):
        """Standard POD / SVD"""
        if self.snapshot_matrix is None: return

        # Center data
        self.X_mean = np.mean(self.snapshot_matrix, axis=1, keepdims=True)
        X_centered = self.snapshot_matrix - self.X_mean
        
        # SVD
        self.U, self.S, self.Vt = svd(X_centered, full_matrices=False)
        
        # Energy Plot
        energy = (self.S ** 2) / np.sum(self.S ** 2)
        print(f"Top 5 Modes Energy: {energy[:5]*100}")

    def analyze_spectral_coverage(self, k_modes=POD_MODES_TO_KEEP):
        """
        reconstructs the signal using only 'k' modes and compares the 
        FFT of the reconstruction vs. the original signal.
        """
        if self.U is None: return

        print(f"\n--- Analyzing Spectral Coverage with {k_modes} Modes ---")
        
        # Pick a random operating point (snapshot) to test
        test_idx = 0 
        original_signal = self.snapshot_matrix[:, test_idx]
        
        # 1. Reconstruct Signal using k modes
        # X_rec = Mean + U_k * Sigma_k * V_k.T
        U_k = self.U[:, :k_modes]
        S_k = np.diag(self.S[:k_modes])
        Vt_k = self.Vt[:k_modes, :]
        
        # Project specific snapshot back
        # Coefficient for this specific snapshot: a = U^T * (x - mean)
        coeffs = np.dot(U_k.T, (original_signal - self.X_mean.flatten()))
        reconstructed_signal = self.X_mean.flatten() + np.dot(U_k, coeffs)

        # 2. Compute FFT for both
        N = len(self.common_angle)
        dt = (self.common_angle[1] - self.common_angle[0]) # Sampling interval in degrees
        
        # Use rfft (real FFT)
        yf_orig = rfft(original_signal)
        yf_rec  = rfft(reconstructed_signal)
        xf = rfftfreq(N, dt)

        mag_orig = np.abs(yf_orig)
        mag_rec  = np.abs(yf_rec)

        # 3. Compute Spectral Error Metric (Normalized L2 diff per harmonic)
        # We focus on the first 100 harmonics as they dominate e-machines
        max_harmonic_idx = 100 
        
        error_spectrum = np.abs(mag_orig - mag_rec)
        
        # --- VISUALIZATION ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Time Domain Reconstruction
        ax1 = axes[0]
        ax1.plot(self.common_angle, original_signal, 'k-', label='Original CFD/FEA', alpha=0.6)
        ax1.plot(self.common_angle, reconstructed_signal, 'r--', label=f'POD Reconstruction ({k_modes} modes)')
        ax1.set_title(f"Time Domain Check (Op Point: {self.operating_conditions[test_idx]})")
        ax1.set_xlabel("Electrical Angle (deg)")
        ax1.set_ylabel("Force (N)")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Frequency Domain Coverage
        
        ax2 = axes[1]
        
        # Plot Original Spectrum
        ax2.stem(xf[:max_harmonic_idx], mag_orig[:max_harmonic_idx], 
                 linefmt='k-', markerfmt='ko', basefmt=" ", label='Original Spectrum')
        
        # Plot Reconstruction Spectrum (Shifted slightly for visibility)
        ax2.stem(xf[:max_harmonic_idx] + 0.05, mag_rec[:max_harmonic_idx], 
                 linefmt='r-', markerfmt='rx', basefmt=" ", label='Reconstructed Spectrum')

        ax2.set_title(f"Spectral Coverage (FFT) - Do {k_modes} modes capture the harmonics?")
        ax2.set_xlabel("Spatial Order (Harmonic)")
        ax2.set_ylabel("Amplitude")
        ax2.set_yscale('log') # Log scale is crucial for harmonics!
        ax2.legend()
        ax2.grid(True, which="both", ls="-", alpha=0.5)

        plt.tight_layout()
        plt.show()

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Ensure dummy data exists (same as before)
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        # ... (Dummy data generation code from previous response) ...
        # (Assuming data is generated or present)

    analyzer = EmagForceAnalyzer(DATA_FOLDER)
    analyzer.load_and_process_data()
    analyzer.perform_pod_analysis()
    
    # NEW STEP: Check if 5 modes are enough to capture the high-frequency content
    analyzer.analyze_spectral_coverage(k_modes=5)