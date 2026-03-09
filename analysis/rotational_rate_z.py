import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader

# =============== Config ===============
BAG_PATH   = Path("/home/fabri/EECE5554/lab4/data/circle_walk_bag_1")
TOPIC      = "/imu"      # change if needed
BIAS_SECS  = 2.0         # seconds at start assumed stationary (for bias estimate)
PLOT_DEG   = True        # show θz and heading in degrees
# ======================================

def detrend_and_integrate(omega, t, bias_secs=2.0):
    """omega [rad/s], t [s] (monotonic, not necessarily uniform)."""
    omega = np.asarray(omega, float)
    t = np.asarray(t, float)
    idx = np.argsort(t)
    t = t[idx]
    omega = omega[idx]
    # estimate DC bias from the first bias_secs
    mask = (t - t[0]) <= bias_secs
    if not np.any(mask):
        mask = slice(0, min(len(omega), 200))
    bias = float(np.median(omega[mask]))
    omega_detr = omega - bias
    # integrate with trapezoid rule using true dt
    theta = np.zeros_like(omega_detr)
    if len(omega_detr) >= 2:
        dt = np.diff(t)
        theta[1:] = np.cumsum(0.5 * (omega_detr[1:] + omega_detr[:-1]) * dt)
    return omega_detr, theta, bias, t

def heading_from_mag(mx, my, unwrap=True):
    """
    Compute heading ψ from magnetometer (no tilt compensation):
    ψ = atan2(E, N) = atan2(mx, my).
    Returns radians; unwrap for continuity if desired.
    """
    psi = np.arctan2(mx, my)  # [-pi, pi]
    if unwrap:
        psi = np.unwrap(psi)
    return psi

# -------- Read from bag --------
timestamps = []
gz = []
mag_x = []
mag_y = []

with AnyReader([BAG_PATH]) as reader:
    conns = [c for c in reader.connections if c.topic == TOPIC]
    if not conns:
        raise RuntimeError(f"Topic {TOPIC} not found in bag. Available: {[c.topic for c in reader.connections]}")
    for conn, t_ns, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)

        # Gyro Z (rad/s) — per ROS Imu spec
        gz.append(float(msg.imu.angular_velocity.z))

        # Magnetometer: try common layouts
        if hasattr(msg, "mag_field") and hasattr(msg.mag_field, "magnetic_field"):
            mx = float(msg.mag_field.magnetic_field.x)
            my = float(msg.mag_field.magnetic_field.y)
        elif hasattr(msg, "magnetic_field"):
            mx = float(msg.magnetic_field.x)
            my = float(msg.magnetic_field.y)
        else:
            # If your /imu message doesn't carry mag, pick the correct topic and re-run
            raise RuntimeError("Magnetometer fields not found on this message. Use a topic that contains mag data.")
        mag_x.append(mx)
        mag_y.append(my)

        timestamps.append(t_ns * 1e-9)  # ns -> s

# Arrays & zero start time
t = np.asarray(timestamps, float)
t -= t[0]
gz = np.asarray(gz, float)
mag_x = np.asarray(mag_x, float)
mag_y = np.asarray(mag_y, float)

# Effective sample rate (info)
if len(t) > 1:
    fs_est = 1.0 / np.mean(np.diff(np.sort(t)))
    print(f"Estimated sample rate: {fs_est:.2f} Hz")

# --- Gyro Z: detrend + integrate ---
omega_z_detr, theta_z, bias_z, t_sorted = detrend_and_integrate(gz, t, bias_secs=BIAS_SECS)
print(f"Estimated gyro Z bias: {bias_z:.6f} rad/s")

# --- Mag heading ---
psi_mag = heading_from_mag(mag_x, mag_y, unwrap=True)  # radians
psi_mag_plot = np.degrees(psi_mag) if PLOT_DEG else psi_mag
theta_z_plot = np.degrees(theta_z) if PLOT_DEG else theta_z

# --- Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# (1) Rotational rate ωz
ax1.plot(t_sorted, gz[np.argsort(t)], lw=1.0, alpha=0.8, label='ωz raw (rad/s)')
#ax1.plot(t_sorted, omega_z_detr, lw=1.0, alpha=0.9, label='ωz detrended (rad/s)')
ax1.axhline(0, color='k', lw=0.8)
ax1.set_ylabel('ωz (rad/s)')
ax1.set_title('Gyro Z and Magnetic Heading')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# (2) Integrated angle θz
ax2.plot(t_sorted, theta_z_plot, lw=1.2, label='θz (integrated)')
ax2.set_ylabel('θz (deg)' if PLOT_DEG else 'θz (rad)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# (3) Magnetic heading ψ_mag = atan2(E, N) = atan2(mag_x, mag_y)
ax3.plot(t, psi_mag_plot, lw=1.2, label='ψ_mag (from mag)')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Heading (deg)' if PLOT_DEG else 'Heading (rad)')
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend()

plt.tight_layout()
plt.show()

