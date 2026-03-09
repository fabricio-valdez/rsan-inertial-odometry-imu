import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============== Config ===============
CSV_PATH  = Path("/home/fabri/EECE5554/lab4/data/square_walk.csv")
BIAS_SECS = 2.0      # seconds at start assumed stationary (for bias estimate)
PLOT_DEG  = True     # show θz and heading in degrees
# ======================================

# Fixed column indices from your CSV:
SEC_COL, NSEC_COL = 0, 1
WZ_COL            = 21        # gyro z [rad/s]
MX_COL, MY_COL    = 46, 47    # magnetometer x/y [Tesla]

def detrend_and_integrate(omega, t, bias_secs=2.0):
    """omega [rad/s], t [s] (not necessarily uniform)."""
    omega = np.asarray(omega, float)
    t     = np.asarray(t, float)

    # sort by time (safety)
    idx = np.argsort(t)
    t_sorted = t[idx]
    omega    = omega[idx]

    # estimate DC bias from the first bias_secs
    mask = (t_sorted - t_sorted[0]) <= bias_secs
    if not np.any(mask):
        mask = slice(0, min(len(omega), 200))
    bias = float(np.median(omega[mask]))
    omega_detr = omega - bias

    # integrate with trapezoid rule using true dt
    theta = np.zeros_like(omega_detr)
    if len(omega_detr) >= 2:
        dt = np.diff(t_sorted)
        theta[1:] = np.cumsum(0.5 * (omega_detr[1:] + omega_detr[:-1]) * dt)
    return omega_detr, theta, bias, t_sorted

def heading_from_mag(mx, my, unwrap=True):
    """
    ψ = atan2(E, N) = atan2(mx, my). No tilt compensation.
    Returns radians; unwrap for continuity if desired.
    """
    psi = np.arctan2(mx, my)  # [-pi, pi]
    if unwrap:
        psi = np.unwrap(psi)
    return psi

# -------- Read CSV --------
df = pd.read_csv(
    CSV_PATH,
    header=None,
    sep=None,
    engine="python",
    on_bad_lines="skip",
    encoding_errors="ignore"
)

# Build time (seconds, start at zero)
sec  = pd.to_numeric(df.iloc[:, SEC_COL],  errors="coerce").to_numpy(float)
nsec = pd.to_numeric(df.iloc[:, NSEC_COL], errors="coerce").to_numpy(float)
t = sec + nsec * 1e-9
t -= np.nanmin(t)

# Extract signals
gz = pd.to_numeric(df.iloc[:, WZ_COL], errors="coerce").to_numpy(float)   # rad/s
mx = pd.to_numeric(df.iloc[:, MX_COL], errors="coerce").to_numpy(float)   # Tesla
my = pd.to_numeric(df.iloc[:, MY_COL], errors="coerce").to_numpy(float)   # Tesla

# Keep only finite rows
valid = np.isfinite(t) & np.isfinite(gz) & np.isfinite(mx) & np.isfinite(my)
t, gz, mx, my = [x[valid] for x in (t, gz, mx, my)]

# Effective sample rate (info)
if len(t) > 1:
    fs_est = 1.0 / np.mean(np.diff(np.sort(t)))
    print(f"Estimated sample rate: {fs_est:.2f} Hz")

# --- Gyro Z: detrend + integrate ---
omega_z_detr, theta_z, bias_z, t_sorted = detrend_and_integrate(gz, t, bias_secs=BIAS_SECS)
print(f"Estimated gyro Z bias: {bias_z:.6f} rad/s")

# --- Magnetic heading (radians; unwrapped) ---
psi_mag = heading_from_mag(mx, my, unwrap=True)
psi_mag_plot  = np.degrees(psi_mag) if PLOT_DEG else psi_mag
theta_z_plot  = np.degrees(theta_z) if PLOT_DEG else theta_z

# --- Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# (1) Rotational rate ωz
ax1.plot(t_sorted, gz[np.argsort(t)], lw=1.0, alpha=0.85, label='ωz raw (rad/s)')
# ax1.plot(t_sorted, omega_z_detr, lw=1.0, alpha=0.9, label='ωz detrended (rad/s)')
ax1.axhline(0, color='k', lw=0.8)
ax1.set_ylabel('ωz (rad/s)')
ax1.set_title('Gyro Z and Magnetic Heading — CSV input')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# (2) Integrated angle θz
ax2.plot(t_sorted, theta_z_plot, lw=1.2, label='θz (integrated)')
ax2.set_ylabel('θz (deg)' if PLOT_DEG else 'θz (rad)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# (3) Magnetic heading ψ_mag = atan2(E, N) = atan2(mx, my)
ax3.plot(t, psi_mag_plot, lw=1.2, label='ψ_mag (from mag)')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Heading (deg)' if PLOT_DEG else 'Heading (rad)')
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend()

plt.tight_layout()
plt.show()
