import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ================== Config ==================
CSV_PATH  = Path("/home/fabri/EECE5554/lab4/data/square_walk.csv")
BIAS_SECS = 2.0        # seconds from start assumed stationary for bias estimate
PLOT_DEG  = False      # if True, plot angle in degrees
# ============================================

# Fixed column indices from your CSV structure:
SEC_COL, NSEC_COL = 0, 1
GYRO_COL_MAP = {"x": 19, "y": 20}  # wz=21 if you ever need Z

def detrend_and_integrate(omega, t, bias_secs=2.0):
    """omega: rad/s, t: seconds (monotonic not required).
       Returns: omega_detr, theta, bias, t_sorted
    """
    omega = np.asarray(omega, float)
    t = np.asarray(t, float)

    # sort by time (safety)
    idx = np.argsort(t)
    t_sorted = t[idx]
    omega = omega[idx]

    # estimate bias from the first bias_secs
    mask = (t_sorted - t_sorted[0]) <= bias_secs
    if not np.any(mask):
        mask = slice(0, min(len(omega), 200))  # fallback
    bias = float(np.median(omega[mask]))
    omega_detr = omega - bias

    # trapezoidal integration with true dt
    theta = np.zeros_like(omega_detr)
    if len(omega_detr) >= 2:
        dt = np.diff(t_sorted)
        theta[1:] = np.cumsum(0.5 * (omega_detr[1:] + omega_detr[:-1]) * dt)
    return omega_detr, theta, bias, t_sorted

def pick_axis_xy():
    ax = input("Choose gyro axis to plot (x / y): ").strip().lower()
    if ax not in {"x", "y"}:
        raise ValueError("Axis must be 'x' or 'y'.")
    return ax

# -------- Read CSV --------
df = pd.read_csv(
    CSV_PATH,
    header=None,
    sep=None,
    engine="python",
    on_bad_lines="skip",
    encoding_errors="ignore"
)

# Build timestamps (seconds, start at zero)
sec  = pd.to_numeric(df.iloc[:, SEC_COL], errors="coerce").to_numpy(float)
nsec = pd.to_numeric(df.iloc[:, NSEC_COL], errors="coerce").to_numpy(float)
t = sec + nsec * 1e-9
t -= np.nanmin(t)

axis = pick_axis_xy()
gyro_col = GYRO_COL_MAP[axis]

omega = pd.to_numeric(df.iloc[:, gyro_col], errors="coerce").to_numpy(float)

# Keep only finite rows
valid = np.isfinite(t) & np.isfinite(omega)
t = t[valid]
omega = omega[valid]

# (optional) print effective sample rate
if len(t) > 1:
    fs_est = 1.0 / np.mean(np.diff(np.sort(t)))
    print(f"Estimated sample rate: {fs_est:.2f} Hz")

# -------- Detrend & integrate --------
omega_detr, theta, bias, t_sorted = detrend_and_integrate(omega, t, bias_secs=BIAS_SECS)
print(f"Estimated gyro {axis.upper()} bias: {bias:.6f} rad/s")

# -------- Plot: rate (rad/s) and angle (rad or deg) --------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Rotational rate
ax1.plot(t_sorted, omega, lw=1.0, alpha=0.8, label=f'ω{axis} raw (rad/s)')
# ax1.plot(t_sorted, omega_detr, lw=1.0, alpha=0.9, label=f'ω{axis} detrended (rad/s)')
ax1.axhline(0, color='k', lw=0.8)
ax1.set_ylabel('Angular rate (rad/s)')
ax1.set_title(f'Gyro {axis.upper()} — Rotational Rate and Integrated Angle')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Integrated rotation
if PLOT_DEG:
    ax2.plot(t_sorted, np.degrees(theta), lw=1.2, label=f'θ{axis} (deg)')
    ax2.set_ylabel('Angle (deg)')
else:
    ax2.plot(t_sorted, theta, lw=1.2, label=f'θ{axis} (rad)')
    ax2.set_ylabel('Angle (rad)')

ax2.set_xlabel('Time (s)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()
