import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============== Config ==============
CSV_PATH  = Path("/home/fabri/EECE5554/lab4/data/square_walk.csv")
CSV_PATH  = Path("/home/fabri/EECE5554/lab5/data/driving1")

BIAS_SECS = 2.0           # seconds assumed stationary at start for bias estimate
# ====================================

# Fixed column indices from your CSV:
SEC_COL, NSEC_COL = 0, 1
ACC_COLS = {"x": 31, "y": 32, "z": 33}

def detrend_and_integrate(acc, t, bias_secs=2.0):
    """
    acc: 1D array (m/s^2)
    t:   1D array (s), not necessarily uniform
    Returns:
      acc_detr (m/s^2), vel (m/s), bias (m/s^2), t_sorted
    """
    acc = np.asarray(acc, float)
    t   = np.asarray(t, float)

    # sort by time (safety)
    idx = np.argsort(t)
    t_sorted = t[idx]
    acc_sorted = acc[idx]

    # estimate DC bias using first 'bias_secs' seconds
    mask = (t_sorted - t_sorted[0]) <= bias_secs
    if not np.any(mask):
        mask = slice(0, min(len(acc_sorted), 200))  # fallback if timestamps are weird
    bias = float(np.median(acc_sorted[mask]))
    acc_detr = acc_sorted - bias

    # integrate with trapezoid rule using true dt
    vel = np.zeros_like(acc_detr)
    if len(acc_detr) >= 2:
        dt = np.diff(t_sorted)
        vel[1:] = np.cumsum(0.5 * (acc_detr[1:] + acc_detr[:-1]) * dt)

    return acc_detr, vel, bias, t_sorted

def pick_axis():
    ax = input("Choose axis to plot (x / y / z): ").strip().lower()
    if ax not in {"x", "y", "z"}:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    return ax

def main():
    axis = pick_axis()
    print(f"[INFO] Reading acceleration on axis '{axis.upper()}' from CSV")

    # Read CSV robustly (no header)
    df = pd.read_csv(
        CSV_PATH,
        header=None,
        sep=None,
        engine="python",
        on_bad_lines="skip",
        encoding_errors="ignore"
    )

    # Build timestamps (seconds, start at zero)
    sec  = pd.to_numeric(df.iloc[:, SEC_COL],  errors="coerce").to_numpy(float)
    nsec = pd.to_numeric(df.iloc[:, NSEC_COL], errors="coerce").to_numpy(float)
    t = sec + nsec * 1e-9
    t -= np.nanmin(t)

    # Select accel column
    acc_col = ACC_COLS[axis]
    acc = pd.to_numeric(df.iloc[:, acc_col], errors="coerce").to_numpy(float)

    # Keep only finite rows
    valid = np.isfinite(t) & np.isfinite(acc)
    t = t[valid]
    acc = acc[valid]

    # Info only: effective sample rate
    if len(t) > 1:
        fs_est = 1.0 / np.mean(np.diff(np.sort(t)))
        print(f"[INFO] Estimated sample rate: {fs_est:.2f} Hz")

    # Detrend & integrate
    acc_detr, vel, bias_est, t_sorted = detrend_and_integrate(acc, t, bias_secs=BIAS_SECS)
    print(f"[INFO] Estimated accel {axis.upper()} bias: {bias_est:.6f} m/s^2")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t_sorted, acc[np.argsort(t)], lw=1.0, alpha=0.85, label=f'a{axis} raw')
    # ax1.plot(t_sorted, acc_detr, lw=1.0, alpha=0.9, label=f'a{axis} detrended')
    ax1.axhline(0, color='k', lw=0.8)
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title(f'Acceleration and Velocity on {axis.upper()} axis — CSV input')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.plot(t_sorted, vel, lw=1.2, label=f'v{axis} (integrated)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
