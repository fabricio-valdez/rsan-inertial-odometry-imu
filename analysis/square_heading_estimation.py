import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============ Config ============
CSV_PATH   = Path("/home/fabri/EECE5554/lab4/data/square_walk.csv")
BIAS_SECS  = 2.0          # seconds assumed stationary at start
USE_DEGREES_PLOT = True   # only affects labels elsewhere; math stays in radians
# ===============================

# Fixed column indices from your CSV:
SEC_COL, NSEC_COL = 0, 1
AX_COL            = 31            # linear acceleration X [m/s^2]
WZ_COL            = 21            # gyro Z [rad/s]
MX_COL, MY_COL, MZ_COL = 46, 47, 48  # magnetometer [Tesla]

# --- Simple mag calibration (your min-max) ---
def calibrate_mag_minmax(mx, my, mz):
    M = np.column_stack([mx, my, mz]).astype(float)
    mins = np.nanmin(M, axis=0); maxs = np.nanmax(M, axis=0)
    b = (maxs + mins) / 2.0
    M_bias = M - b
    sx, sy, sz = (np.nanmax(M_bias, axis=0) - np.nanmin(M_bias, axis=0)) / 2.0
    eps = 1e-9
    sx = sx if sx > eps else eps; sy = sy if sy > eps else eps; sz = sz if sz > eps else eps
    s_avg = (sx + sy + sz) / 3.0
    S = np.diag([s_avg/sx, s_avg/sy, s_avg/sz])
    M_cal = M_bias @ S
    return M_cal, b, S

def estimate_bias(x, t, bias_secs):
    t = np.asarray(t); x = np.asarray(x)
    m = (t - t[0]) <= bias_secs
    if not np.any(m):
        m = slice(0, min(len(x), 200))
    return float(np.median(x[m]))

def trapz_integrate(y, t):
    """cumulative trapezoid integral with true (possibly nonuniform) dt. Returns array same length."""
    y = np.asarray(y); t = np.asarray(t)
    out = np.zeros_like(y, dtype=float)
    if len(y) >= 2:
        dt = np.diff(t)
        out[1:] = np.cumsum(0.5*(y[1:] + y[:-1]) * dt)
    return out

def heading_from_mag(mx_cal, my_cal):
    """ψ = atan2(E,N) = atan2(mx, my), unwrap for continuity."""
    psi = np.arctan2(mx_cal, my_cal)
    return np.unwrap(psi)

# -------- Read from CSV --------
df = pd.read_csv(
    CSV_PATH,
    header=None,
    sep=None,
    engine="python",
    on_bad_lines="skip",
    encoding_errors="ignore"
)

# Arrays & time normalization
sec  = pd.to_numeric(df.iloc[:, SEC_COL],  errors="coerce").to_numpy(float)
nsec = pd.to_numeric(df.iloc[:, NSEC_COL], errors="coerce").to_numpy(float)
t = sec + nsec * 1e-9
t -= np.nanmin(t)

ax = pd.to_numeric(df.iloc[:, AX_COL], errors="coerce").to_numpy(float)   # m/s^2
gz = pd.to_numeric(df.iloc[:, WZ_COL], errors="coerce").to_numpy(float)   # rad/s
mx = pd.to_numeric(df.iloc[:, MX_COL], errors="coerce").to_numpy(float)   # Tesla
my = pd.to_numeric(df.iloc[:, MY_COL], errors="coerce").to_numpy(float)   # Tesla
mz = pd.to_numeric(df.iloc[:, MZ_COL], errors="coerce").to_numpy(float)   # Tesla

# Keep only finite rows
valid = np.isfinite(t) & np.isfinite(ax) & np.isfinite(gz) & np.isfinite(mx) & np.isfinite(my) & np.isfinite(mz)
t, ax, gz, mx, my, mz = [x[valid] for x in (t, ax, gz, mx, my, mz)]

# Sort by time (safety)
idx = np.argsort(t)
t  = t[idx]; ax = ax[idx]; gz = gz[idx]; mx = mx[idx]; my = my[idx]; mz = mz[idx]

# Info: estimated sample rate
if len(t) > 1:
    fs_est = 1.0/np.mean(np.diff(t))
    print(f"[INFO] Estimated sample rate: {fs_est:.2f} Hz")

# --- Detrend accel X (remove small DC) ---
bias_ax = estimate_bias(ax, t, BIAS_SECS)
ax_detr = ax - bias_ax
print(f"[INFO] Accel X bias ~ {bias_ax:.6f} m/s^2")

# --- Detrend gyro Z and integrate to heading ---
bias_gz = estimate_bias(gz, t, BIAS_SECS)
gz_detr = gz - bias_gz
psi_gyro = trapz_integrate(gz_detr, t)     # radians
psi_gyro = np.unwrap(psi_gyro)
print(f"[INFO] Gyro Z bias ~ {bias_gz:.6f} rad/s")

# --- Calibrate mag & compute heading from mag ---
# (Units can stay in Tesla; scaling cancels in atan2. Calibration helps ellipse->circle.)
M_cal, b_mag, S_mag = calibrate_mag_minmax(mx, my, mz)
mx_cal, my_cal = M_cal[:,0], M_cal[:,1]
psi_mag = heading_from_mag(mx_cal, my_cal)  # radians

# --- Project body-forward acceleration into N/E using each heading ---
# a_N = ax_detr * cos(psi), a_E = ax_detr * sin(psi)
aN_mag  = ax_detr * np.cos(psi_mag)
aE_mag  = ax_detr * np.sin(psi_mag)
aN_gyro = ax_detr * np.cos(psi_gyro)
aE_gyro = ax_detr * np.sin(psi_gyro)

# --- Twice integrate to velocity and position (assumes initial v,p = 0) ---
vN_mag  = trapz_integrate(aN_mag, t);  vE_mag  = trapz_integrate(aE_mag, t)
pN_mag  = trapz_integrate(vN_mag, t);  pE_mag  = trapz_integrate(vE_mag, t)

vN_gyro = trapz_integrate(aN_gyro, t); vE_gyro = trapz_integrate(aE_gyro, t)
pN_gyro = trapz_integrate(vN_gyro, t); pE_gyro = trapz_integrate(vE_gyro, t)

# --- Plot N vs E for both headings ---
plt.figure(figsize=(7,7))
plt.plot(pE_mag,  pN_mag,  '.', markersize=3, label='Mag heading (NE from ax)')
plt.plot(pE_gyro, pN_gyro, '.', markersize=3, label='Gyro heading (NE from ax)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('E (m)')
plt.ylabel('N (m)')
plt.title('N vs E position from ax(t) with Mag vs Gyro heading — CSV input')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
