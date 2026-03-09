import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("/home/fabri/EECE5554/lab4/data/square_walk.csv")

# Read CSV with no header; skip any malformed lines
df = pd.read_csv(CSV_PATH, header=None, sep=None, engine="python",
                 on_bad_lines="skip", encoding_errors="ignore")

# --- Fixed column indices from your file ---
SEC_COL, NSEC_COL = 0, 1
WX_COL,  WY_COL,  WZ_COL  = 19, 20, 21      # rad/s
AX_COL,  AY_COL,  AZ_COL  = 31, 32, 33      # m/s^2
MX_COL,  MY_COL,  MZ_COL  = 46, 47, 48      # Tesla

# --- Build time (seconds, start at 0) ---
sec  = pd.to_numeric(df.iloc[:, SEC_COL], errors="coerce").to_numpy(float)
nsec = pd.to_numeric(df.iloc[:, NSEC_COL], errors="coerce").to_numpy(float)
t = sec + nsec*1e-9
t -= np.nanmin(t)

# --- Extract sensors (numeric, drop non-finite rows once) ---
wx = pd.to_numeric(df.iloc[:, WZ_COL-2], errors="coerce").to_numpy(float)  # wx
wy = pd.to_numeric(df.iloc[:, WZ_COL-1], errors="coerce").to_numpy(float)  # wy
wz = pd.to_numeric(df.iloc[:, WZ_COL],   errors="coerce").to_numpy(float)  # wz

ax = pd.to_numeric(df.iloc[:, AX_COL], errors="coerce").to_numpy(float)
ay = pd.to_numeric(df.iloc[:, AY_COL], errors="coerce").to_numpy(float)
az = pd.to_numeric(df.iloc[:, AZ_COL], errors="coerce").to_numpy(float)

mx = pd.to_numeric(df.iloc[:, MX_COL], errors="coerce").to_numpy(float)
my = pd.to_numeric(df.iloc[:, MY_COL], errors="coerce").to_numpy(float)
mz = pd.to_numeric(df.iloc[:, MZ_COL], errors="coerce").to_numpy(float)

mask = np.isfinite(t) & np.isfinite(wx) & np.isfinite(ax) & np.isfinite(mx) & np.isfinite(my) & np.isfinite(mz)
t, wx, wy, wz, ax, ay, az, mx, my, mz = [x[mask] for x in (t, wx, wy, wz, ax, ay, az, mx, my, mz)]

# mag Tesla -> microTesla for nicer plotting
mx_uT, my_uT, mz_uT = mx*1e6, my*1e6, mz*1e6

# --- Simple mag min–max calibration (same as you used) ---
def calibrate_mag_minmax(mx, my, mz):
    M = np.column_stack([mx, my, mz]).astype(float)
    mins = np.nanmin(M, axis=0); maxs = np.nanmax(M, axis=0)
    b = (maxs + mins)/2.0
    M_bias = M - b
    half_ranges = (np.nanmax(M_bias, axis=0) - np.nanmin(M_bias, axis=0))/2.0
    half_ranges = np.maximum(half_ranges, 1e-9)
    s_avg = np.mean(half_ranges)
    S = np.diag(s_avg / half_ranges)
    M_cal = M_bias @ S
    return M_cal, b, S

M_cal_uT, b_uT, S = calibrate_mag_minmax(mx_uT, my_uT, mz_uT)
E_raw, N_raw = mx_uT, my_uT
E_cal, N_cal = M_cal_uT[:,0], M_cal_uT[:,1]

# --- Plot raw vs calibrated N–E ---
plt.figure(figsize=(7,7))
plt.scatter(E_raw, N_raw, s=5, alpha=0.6, label='Uncalibrated', color='steelblue')
plt.scatter(E_cal, N_cal, s=5, alpha=0.6, label='Calibrated', color='orange')
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Magnetometer N–E (Raw vs Calibrated) — CSV (fixed column map)")
plt.xlabel("Easting (µT)")
plt.ylabel("Northing (µT)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

print("Mag bias b [µT]:", b_uT)
print("Diag scale S:\n", S)
