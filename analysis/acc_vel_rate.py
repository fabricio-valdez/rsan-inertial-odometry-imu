import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader

# ============== Config ==============
BAG_PATH  = Path("/home/fabri/EECE5554/lab4/data/circle_walk_bag_1")
TOPIC     = "/imu"        # change if your accel lives on a different topic
BIAS_SECS = 2.0           # seconds assumed stationary at start for bias estimate
# ====================================

def detrend_and_integrate(acc, t, bias_secs=2.0):
    """
    acc: 1D array (m/s^2)
    t:   1D array (s), monotonic (not necessarily uniform)
    Returns:
      acc_detr (m/s^2), vel (m/s), bias (m/s^2), t_sorted
    """
    acc = np.asarray(acc, float)
    t   = np.asarray(t, float)

    # sort by time (safety)
    idx = np.argsort(t)
    t   = t[idx]
    acc = acc[idx]

    # estimate DC bias using first 'bias_secs' seconds
    mask = (t - t[0]) <= bias_secs
    if not np.any(mask):
        mask = slice(0, min(len(acc), 200))  # fallback if timestamps are weird
    bias = float(np.median(acc[mask]))
    acc_detr = acc - bias

    # integrate with trapezoid rule using true dt
    vel = np.zeros_like(acc_detr)
    if len(acc_detr) >= 2:
        dt = np.diff(t)
        vel[1:] = np.cumsum(0.5 * (acc_detr[1:] + acc_detr[:-1]) * dt)

    return acc_detr, vel, bias, t

def pick_axis():
    ax = input("Choose axis to plot (x / y / z): ").strip().lower()
    if ax not in {"x", "y", "z"}:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    return ax

def main():
    axis = pick_axis()
    print(f"[INFO] Reading acceleration on axis '{axis.upper()}' from {TOPIC}")

    timestamps = []
    ax_vals, ay_vals, az_vals = [], [], []

    with AnyReader([BAG_PATH]) as reader:
        conns = [c for c in reader.connections if c.topic == TOPIC]
        if not conns:
            raise RuntimeError(f"Topic '{TOPIC}' not found. Available: {[c.topic for c in reader.connections]}")
        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)

            # sensor_msgs/Imu layout (m/s^2)
            ax_vals.append(float(msg.imu.linear_acceleration.x))
            ay_vals.append(float(msg.imu.linear_acceleration.y))
            az_vals.append(float(msg.imu.linear_acceleration.z))
            timestamps.append(t_ns * 1e-9)  # ns -> s

    t = np.asarray(timestamps, float)
    t -= t[0]  # start at zero

    ax_map = {"x": np.asarray(ax_vals), "y": np.asarray(ay_vals), "z": np.asarray(az_vals)}
    acc_sel = ax_map[axis]

    # Info only: effective sample rate
    if len(t) > 1:
        fs_est = 1.0 / np.mean(np.diff(np.sort(t)))
        print(f"[INFO] Estimated sample rate: {fs_est:.2f} Hz")

    # Detrend & integrate
    acc_detr, vel, bias_est, t_sorted = detrend_and_integrate(acc_sel, t, bias_secs=BIAS_SECS)
    print(f"[INFO] Estimated accel {axis.upper()} bias: {bias_est:.6f} m/s^2")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t_sorted, acc_sel[np.argsort(t)], lw=1.0, alpha=0.85, label=f'a{axis} raw')
    #ax1.plot(t_sorted, acc_detr, lw=1.0, alpha=0.9, label=f'a{axis} detrended')
    ax1.axhline(0, color='k', lw=0.8)
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title(f'Acceleration and Velocity on {axis.upper()} axis')
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

