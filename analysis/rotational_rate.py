import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader

# ================== Config ==================
BAG_PATH   = Path("/home/fabri/EECE5554/lab4/data/circle_walk_bag_1")
TOPIC      = "/imu"     # change if your gyro lives on a different topic
BIAS_SECS  = 2.0        # seconds from start assumed stationary for bias estimate
AXIS       = "y"        # 'x' | 'y' | 'z'  (here you asked for X)
PLOT_DEG   = False      # if True, plot angle in degrees
# ============================================

def detrend_and_integrate(omega, t, bias_secs=2.0):
    """omega: rad/s, t: seconds (monotonic, not necessarily uniform)."""
    omega = np.asarray(omega, float)
    t = np.asarray(t, float)

    # sort by time (safety)
    idx = np.argsort(t)
    t = t[idx]
    omega = omega[idx]

    # estimate bias from the first bias_secs
    mask = (t - t[0]) <= bias_secs
    if not np.any(mask):
        mask = slice(0, min(len(omega), 200))  # fallback
    bias = np.median(omega[mask])
    omega_detr = omega - bias

    # trapezoidal integration with true dt
    theta = np.zeros_like(omega_detr)
    if len(omega_detr) >= 2:
        dt = np.diff(t)
        theta[1:] = np.cumsum(0.5 * (omega_detr[1:] + omega_detr[:-1]) * dt)
    return omega_detr, theta, bias

# -------- Read gyro X (rad/s) + timestamps (s) from the bag --------
timestamps = []
gx = []

with AnyReader([BAG_PATH]) as reader:
    conns = [c for c in reader.connections if c.topic == TOPIC]
    if not conns:
        raise RuntimeError(f"Topic {TOPIC} not found in bag.")
    for conn, t_ns, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        # VN-100 via sensor_msgs/Imu is rad/s by spec
        gx.append(float(msg.imu.angular_velocity.y))
        timestamps.append(t_ns * 1e-9)  # ns -> s

t = np.asarray(timestamps, float)
t -= t[0]  # start at zero
gx = np.asarray(gx, float)

# (optional) print effective sample rate
if len(t) > 1:
    fs_est = 1.0 / np.mean(np.diff(np.sort(t)))
    print(f"Estimated sample rate: {fs_est:.2f} Hz")

# -------- Detrend & integrate (X axis) --------
omega_x_detr, theta_x, bias_x = detrend_and_integrate(gx, t, bias_secs=BIAS_SECS)
print(f"Estimated gyro X bias: {bias_x:.6f} rad/s")

# -------- Plot: rate (rad/s) and angle (rad or deg) --------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Rotational rate
ax1.plot(t, gx, lw=1.0, alpha=0.8, label='ωy raw (rad/s)')
#ax1.plot(t, omega_x_detr, lw=1.0, alpha=0.9, label='ωx detrended (rad/s)')
ax1.axhline(0, color='k', lw=0.8)
ax1.set_ylabel('Angular rate (rad/s)')
ax1.set_title('Gyro Y — Rotational Rate and Integrated Angle')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Integrated rotation
if PLOT_DEG:
    ax2.plot(t, np.degrees(theta_x), lw=1.2, label='θy (deg)')
    ax2.set_ylabel('Angle (deg)')
else:
    ax2.plot(t, theta_x, lw=1.2, label='θy (rad)')
    ax2.set_ylabel('Angle (rad)')

ax2.set_xlabel('Time (s)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()

