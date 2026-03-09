import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader

# ========= CONFIG =========
BAG_FILE = Path("/home/fabri/EECE5554/lab4/data/square_walk_bag/data_0.mcap")  # point to the .mcap file directly
IMU_TOPIC = "/imu"  # change if needed (use the printed topic list below)
# ==========================

def calibrate_mag_minmax(mx, my, mz):
    M = np.column_stack([mx, my, mz]).astype(float)

    # Hard-iron
    mins = np.nanmin(M, axis=0)
    maxs = np.nanmax(M, axis=0)
    b = (maxs + mins) / 2.0
    M_bias = M - b

    # Soft-iron (diag)
    sx, sy, sz = (np.nanmax(M_bias, axis=0) - np.nanmin(M_bias, axis=0)) / 2.0
    eps = 1e-9
    sx = sx if sx > eps else eps
    sy = sy if sy > eps else eps
    sz = sz if sz > eps else eps
    s_avg = (sx + sy + sz) / 3.0
    S = np.diag([s_avg / sx, s_avg / sy, s_avg / sz])

    M_cal = M_bias @ S
    return M_cal, b, S

# ---- Open MCAP directly (no metadata.yaml needed) ----
with AnyReader([BAG_FILE]) as reader:
    # Show what topics/types are available
    print("Available topics:")
    for c in reader.connections:
        print(f"  {c.topic}  ->  {c.msgtype}")

    # Pick connections for the topic you want
    conns = [c for c in reader.connections if c.topic == IMU_TOPIC]
    if not conns:
        raise RuntimeError(f"Topic {IMU_TOPIC} not found. Choose one from the list above and set IMU_TOPIC.")

    timestamps = []
    mag_x, mag_y, mag_z = [], [], []

    # Read messages
    for conn, t_ns, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)

        # Try common layouts:
        # 1) Plain sensor_msgs/msg/MagneticField
        #    (if your topic is directly MagneticField, not inside Imu)
        if hasattr(msg, "magnetic_field"):  # sensor_msgs/MagneticField
            mx = float(msg.magnetic_field.x)
            my = float(msg.magnetic_field.y)
            mz = float(msg.magnetic_field.z)

        # 2) Nested in a custom message, e.g. msg.mag_field.magnetic_field
        elif hasattr(msg, "mag_field") and hasattr(msg.mag_field, "magnetic_field"):
            mx = float(msg.mag_field.magnetic_field.x)
            my = float(msg.mag_field.magnetic_field.y)
            mz = float(msg.mag_field.magnetic_field.z)

        # 3) Some drivers put mag in 'imu' wrapper — unlikely, but safe check
        elif hasattr(msg, "imu") and hasattr(msg.imu, "magnetic_field"):
            mx = float(msg.imu.magnetic_field.x)
            my = float(msg.imu.magnetic_field.y)
            mz = float(msg.imu.magnetic_field.z)

        else:
            # If your /imu message doesn’t carry mag, you may need a different topic (e.g., '/mag' or '/vn/sensors')
            # Print one example to inspect fields:
            print("Message on", IMU_TOPIC, "does not contain magnetometer fields. One example:", msg)
            raise RuntimeError("Choose a topic that contains magnetometer data (see 'Available topics' printed above).")

        mag_x.append(mx)
        mag_y.append(my)
        mag_z.append(mz)
        timestamps.append(t_ns * 1e-9)  # seconds

# ---- Calibrate & plot N–E raw vs calibrated ----
mag_x = np.asarray(mag_x)
mag_y = np.asarray(mag_y)
mag_z = np.asarray(mag_z)

M_cal, b, S = calibrate_mag_minmax(mag_x, mag_y, mag_z)

E_raw = mag_x
N_raw = mag_y
E_cal = M_cal[:, 0]
N_cal = M_cal[:, 1]

plt.figure(figsize=(7, 7))
plt.scatter(E_raw, N_raw, s=5, alpha=0.6, label='Uncalibrated', color='steelblue')
plt.scatter(E_cal, N_cal, s=5, alpha=0.6, label='Calibrated', color='orange')
plt.title("Magnetometer N–E (Raw vs Calibrated) — direct MCAP read")
plt.xlabel("Easting (µT)")
plt.ylabel("Northing (µT)")
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

print("Bias vector b (µT):", b)
print("Diagonal scale matrix S:\n", S)
