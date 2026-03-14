# Autonomous Drone with Raspberry Pi 4B and Camera-Based Attitude Estimation

This guide explains how to run an autonomous drone using a Raspberry Pi 4B as the companion computer, with a Pixhawk flight controller.
It covers how to use a camera to derive optical-flow-based velocity and attitude data that can supplement or replace a dedicated IMU in GPS-denied environments.

The setup uses [MAVSDK-Python](https://github.com/mavlink/MAVSDK-Python) for high-level autonomous flight commands and [OpenCV](https://opencv.org/) for camera-based state estimation.

## Overview

The architecture has three layers:

1. **Pixhawk flight controller** — runs PX4 firmware, handles low-level stabilisation, motor mixing, and safety.
2. **Raspberry Pi 4B companion computer** — runs the optical-flow pipeline and MAVSDK-based mission script.
3. **Camera** (e.g. Raspberry Pi Camera Module v2 or USB webcam) — provides the video frames used for attitude and velocity estimation.

The companion computer sends `VISION_POSITION_ESTIMATE` MAVLink messages to PX4 so that the EKF2 estimator can fuse the camera-derived position and heading with the onboard IMU.

## Hardware Requirements

| Component                        | Notes                                                  |
| -------------------------------- | ------------------------------------------------------ |
| Pixhawk flight controller        | Any Pixhawk-series board running PX4 v1.14 or later    |
| Raspberry Pi 4B (2 GB RAM min.)  | Running Ubuntu 22.04 LTS (64-bit)                      |
| Camera                           | Pi Camera v2 or compatible USB camera                  |
| microSD card (≥16 GB, Class 10)  | For the RPi OS                                         |
| LiPo battery + power module      | Powers the flight controller and, via BEC, the RPi     |
| FTDI USB-to-serial adapter       | Optional — for USB serial connection to Pixhawk TELEM2 |
| Telemetry radio (optional)       | For ground-station monitoring during flights           |

## Wiring and Connections

### Serial Connection Between Pixhawk and RPi

Connect the Pixhawk `TELEM2` port to the RPi GPIO header using the pin mapping below.
Use 3.3 V logic levels on the RPi side — do **not** connect the 5 V VCC pin from `TELEM2` directly to the RPi GPIO.

| Pixhawk TELEM2 Pin  | RPi GPIO Pin           |
| ------------------- | ---------------------- |
| UART5_TX (pin 2)    | RXD (GPIO 15, pin 10)  |
| UART5_RX (pin 3)    | TXD (GPIO 14, pin 8)   |
| GND (pin 6)         | Ground (pin 6)         |

Alternatively, connect via a USB-to-serial adapter plugged into a free USB port on the RPi (exposed as `/dev/ttyUSB0` or `/dev/ttyACM0`).

### Camera Connection

- **Pi Camera Module v2** — connect to the RPi CSI ribbon-cable connector.
  Enable it with `sudo raspi-config` → **Interface Options** → **Camera**.
- **USB webcam** — plug in and verify the device appears as `/dev/video0`.

### Power

Power the RPi from a dedicated 5 V / 3 A BEC connected to the same battery as the flight controller.
Do not rely on the Pixhawk to supply power to the RPi.

## Raspberry Pi 4B Software Setup

### Install Ubuntu 22.04

Follow the [official Raspberry Pi Ubuntu installation guide](https://ubuntu.com/tutorials/how-to-install-ubuntu-desktop-on-raspberry-pi-4#1-overview) to flash Ubuntu 22.04 LTS onto the microSD card.

### Enable the Serial Port

Run `sudo raspi-config` and navigate to **Interface Options** → **Serial Port**.
Select **No** when asked whether the login shell should use the serial port, then **Yes** to enable the serial hardware interface.
Reboot, then append the following lines to `/boot/firmware/config.txt`:

```sh
enable_uart=1
dtoverlay=disable-bt
```

Reboot again and verify the port is available:

```sh
ls /dev/ttyAMA0
```

### Install Python Dependencies

```sh
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-opencv libopencv-dev \
    python3-numpy libatlas-base-dev
pip3 install mavsdk pymavlink
```

### Verify the MAVLink Connection

Install `mavproxy` to confirm the serial link is working before running the flight script:

```sh
pip3 install mavproxy
sudo mavproxy.py --master=/dev/ttyAMA0 --baudrate 57600
```

You should see heartbeat messages in the terminal.
Press **Ctrl+C** to exit MAVProxy once verified.

## PX4 Configuration

Connect the Pixhawk to a laptop via USB and open *QGroundControl*.
Set the following parameters via **Vehicle Setup > Parameters**:

```ini
MAV_1_CONFIG  = TELEM2       # Enable MAVLink on TELEM2
SER_TEL2_BAUD = 57600        # Match baud rate used in the script
EKF2_EV_CTRL  = 11           # Fuse horizontal position, velocity, and yaw (no vertical vision)
EKF2_HGT_REF  = 1            # Use barometer as height reference (recommended for outdoor flight)
EKF2_AID_MASK = 24           # Vision position + vision yaw fusion
```

Reboot the flight controller after saving the parameters.

::: info
`EKF2_EV_CTRL = 11` enables horizontal position, velocity, and yaw fusion from the external vision source while leaving altitude to the barometer.
Set bit 1 (`EKF2_EV_CTRL = 15`) only if you have a reliable external altitude source such as a downward-facing rangefinder.
Adjust this value if you only want to fuse a subset of the data.
:::

## Camera-Based Attitude Estimation

The script uses the [Lucas-Kanade sparse optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) algorithm to track feature points between consecutive frames.
The pixel displacement of tracked features is converted to angular rates and a heading estimate that are then sent to PX4 as a `VISION_POSITION_ESTIMATE` MAVLink message.

This approach provides:

- **Horizontal velocity** — derived from the mean optical-flow field.
- **Yaw (heading)** — integrated from the rotational component of the flow field.
- **Relative altitude** — derived from the divergence of the flow field when the camera points downward.

::: warning
Camera-only attitude estimation without a barometer or rangefinder is not safe for outdoor flight.
Always use a barometer for altitude and cross-check with a rangefinder when available.
The optical-flow heading estimate drifts over time; it is suitable for short flights or GPS-denied indoor environments only.
:::

## Autonomous Flight Script

Before running the script, start the mavlink-router UDP bridge (see [MAVSDK UDP Bridge Setup](#mavsdk-udp-bridge-setup) below) so that MAVSDK can connect to the flight controller.

Save the script below to the RPi as `autonomous_drone.py`.

```python
#!/usr/bin/env python3
"""
Autonomous drone controller for Raspberry Pi 4B + Pixhawk.

Uses OpenCV Lucas-Kanade optical flow to derive attitude data from a
downward-facing or forward-facing camera, then sends VISION_POSITION_ESTIMATE
messages to PX4 via pymavlink while MAVSDK handles the high-level mission.

Usage:
    python3 autonomous_drone.py [--serial /dev/ttyAMA0] [--baud 57600]
                                [--camera 0] [--altitude 2.5]
                                [--mission-speed 1.5] [--leg-duration 3.0]
"""

import argparse
import math
import threading
import time

import asyncio
import cv2
import numpy as np

from pymavlink import mavutil
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityNedYaw)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Camera intrinsics — replace with values from your camera calibration.
# For a Raspberry Pi Camera Module v2 at 640×480 the focal length is
# approximately 640 pixels. Run OpenCV camera calibration to obtain precise
# values: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
CAMERA_FX = 320.0          # Focal length x (pixels) — calibrate for your camera
CAMERA_FY = 320.0          # Focal length y (pixels)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FLOW_SCALE = 0.001         # Scales pixel flow to approximate velocity (m/s per pixel/frame)
HEADING_SCALE = 0.002      # Scales rotational flow to yaw rate (rad/s)
PUBLISH_HZ = 30            # Rate at which vision estimates are sent to PX4

# Minimum dt to prevent division-by-zero when computing flow-derived rates.
MIN_DT = 1e-4              # seconds

# Number of tracked features below which the detector is re-run.
MIN_FEATURES_REFRESH = 50

# Autonomous mission parameters — adjust to match your flying space.
MISSION_SPEED_MS = 1.5     # Velocity for each leg of the square (m/s)
MISSION_LEG_DURATION_S = 3.0  # Duration of each leg (seconds)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
)


# ---------------------------------------------------------------------------
# Optical-flow state estimator
# ---------------------------------------------------------------------------

class CameraEstimator:
    """Derives velocity and heading from consecutive camera frames."""

    def __init__(self, camera_index: int = 0) -> None:
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._prev_time: float = time.monotonic()

        # Estimated state
        self.vx: float = 0.0      # Forward velocity (m/s, body frame)
        self.vy: float = 0.0      # Lateral velocity (m/s, body frame)
        self.yaw: float = 0.0     # Heading (radians, accumulated)
        self.yaw_rate: float = 0.0

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background capture and estimation thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cap.release()

    def get_state(self) -> dict:
        """Return the latest estimated state as a dict."""
        with self._lock:
            return {
                "vx": self.vx,
                "vy": self.vy,
                "yaw": self.yaw,
                "yaw_rate": self.yaw_rate,
            }

    def _run(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time.monotonic()
            dt = now - self._prev_time
            self._prev_time = now

            if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < 10:
                self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
                self._prev_gray = gray
                continue

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts, None, **LK_PARAMS
            )

            good_old = self._prev_pts[status == 1]
            good_new = next_pts[status == 1]

            if len(good_new) < 4:
                self._prev_pts = None
                self._prev_gray = gray
                continue

            # Mean translational flow (pixels)
            flow = good_new - good_old
            mean_dx = float(np.median(flow[:, 0]))
            mean_dy = float(np.median(flow[:, 1]))

            # Convert to approximate body-frame velocities
            vx = -mean_dy / CAMERA_FX / max(dt, MIN_DT)   # Forward
            vy =  mean_dx / CAMERA_FY / max(dt, MIN_DT)   # Lateral

            # Estimate rotational component via affine transform
            transform, _ = cv2.estimateAffinePartial2D(good_old, good_new)
            yaw_rate = 0.0
            if transform is not None:
                yaw_rate = math.atan2(float(transform[1, 0]), float(transform[0, 0])) / max(dt, MIN_DT)
                yaw_rate *= HEADING_SCALE

            with self._lock:
                self.vx = vx * FLOW_SCALE
                self.vy = vy * FLOW_SCALE
                self.yaw_rate = yaw_rate
                self.yaw += yaw_rate * dt

            # Refresh feature points when count falls below threshold
            if len(good_new) < MIN_FEATURES_REFRESH:
                self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
            else:
                self._prev_pts = good_new.reshape(-1, 1, 2)
            self._prev_gray = gray


# ---------------------------------------------------------------------------
# MAVLink vision-position publisher
# ---------------------------------------------------------------------------

class VisionPublisher:
    """Sends VISION_POSITION_ESTIMATE messages to PX4 via pymavlink."""

    def __init__(self, serial_port: str, baud: int, estimator: CameraEstimator) -> None:
        self._conn = mavutil.mavlink_connection(serial_port, baud=baud)
        self._conn.wait_heartbeat(timeout=10)
        print(f"[VisionPublisher] Heartbeat from system {self._conn.target_system}")
        self._estimator = estimator
        self._running = False
        self._thread: threading.Thread | None = None

        # Accumulated position estimate (metres)
        self._x: float = 0.0
        self._y: float = 0.0
        self._z: float = 0.0
        self._prev_time: float = time.monotonic()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        interval = 1.0 / PUBLISH_HZ
        while self._running:
            state = self._estimator.get_state()
            now = time.monotonic()
            dt = now - self._prev_time
            self._prev_time = now

            # Integrate velocity to position (NED frame)
            yaw = state["yaw"]
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            self._x += (cos_y * state["vx"] - sin_y * state["vy"]) * dt
            self._y += (sin_y * state["vx"] + cos_y * state["vy"]) * dt
            # z is left as 0 — PX4 uses the barometer for altitude (EKF2_HGT_REF = 1).

            roll, pitch = 0.0, 0.0
            usec = int(time.monotonic() * 1e6)
            self._conn.mav.vision_position_estimate_send(
                usec,              # usec — timestamp
                self._x,           # x (metres, NED)
                self._y,           # y (metres, NED)
                self._z,           # z (metres, NED, positive down)
                roll,              # roll (rad)
                pitch,             # pitch (rad)
                yaw,               # yaw (rad)
            )
            time.sleep(interval)


# ---------------------------------------------------------------------------
# Autonomous mission (MAVSDK)
# ---------------------------------------------------------------------------

async def run_mission(target_altitude_m: float, serial_port: str,
                      speed_ms: float = MISSION_SPEED_MS,
                      leg_duration_s: float = MISSION_LEG_DURATION_S) -> None:
    """Arm the drone, take off, fly a square, and land."""

    drone = System()
    # MAVSDK connects to the flight controller via MAVLink over UDP.
    # Start mavlink-routerd before running this script (see the
    # "MAVSDK UDP Bridge Setup" section in the documentation).
    await drone.connect(system_address="udp://:14540")

    print("[Mission] Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[Mission] Drone connected")
            break

    print("[Mission] Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[Mission] Position estimate OK")
            break

    print(f"[Mission] Arming and taking off to {target_altitude_m} m")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(target_altitude_m)
    await drone.action.takeoff()
    await asyncio.sleep(5)

    # Fly a 5 m square in offboard velocity mode
    print("[Mission] Starting offboard square flight")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as err:
        print(f"[Mission] Offboard start failed: {err}")
        await drone.action.land()
        return

    legs = [
        (speed_ms, 0.0, 0.0, 0.0),    # North
        (0.0, speed_ms, 0.0, 90.0),   # East
        (-speed_ms, 0.0, 0.0, 180.0), # South
        (0.0, -speed_ms, 0.0, 270.0), # West
    ]
    for vn, ve, vd, yaw_deg in legs:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, vd, yaw_deg))
        await asyncio.sleep(leg_duration_s)

    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    print("[Mission] Stopping offboard and landing")
    await drone.offboard.stop()
    await drone.action.land()

    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("[Mission] Landed successfully")
            break

    await drone.action.disarm()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="/dev/ttyAMA0",
                        help="Serial port connected to Pixhawk TELEM2 (default: /dev/ttyAMA0)")
    parser.add_argument("--baud", type=int, default=57600,
                        help="Baud rate for serial port (default: 57600)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--altitude", type=float, default=2.5,
                        help="Takeoff altitude in metres (default: 2.5)")
    parser.add_argument("--mission-speed", type=float, default=MISSION_SPEED_MS,
                        help=f"Velocity for each mission leg in m/s (default: {MISSION_SPEED_MS})")
    parser.add_argument("--leg-duration", type=float, default=MISSION_LEG_DURATION_S,
                        help=f"Duration of each mission leg in seconds (default: {MISSION_LEG_DURATION_S})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[Init] Starting camera estimator")
    estimator = CameraEstimator(camera_index=args.camera)
    estimator.start()
    time.sleep(1)  # Allow estimator to warm up

    print(f"[Init] Starting vision publisher on {args.serial} @ {args.baud}")
    publisher = VisionPublisher(
        serial_port=args.serial,
        baud=args.baud,
        estimator=estimator,
    )
    publisher.start()
    time.sleep(2)  # Allow PX4 to start receiving vision data

    try:
        print("[Init] Running autonomous mission")
        asyncio.run(run_mission(args.altitude, args.serial,
                                speed_ms=args.mission_speed,
                                leg_duration_s=args.leg_duration))
    except KeyboardInterrupt:
        print("[Init] Interrupted by user")
    finally:
        publisher.stop()
        estimator.stop()
        print("[Init] Shutdown complete")


if __name__ == "__main__":
    main()
```

### Running the Script

```sh
sudo python3 autonomous_drone.py \
    --serial /dev/ttyAMA0 \
    --baud 57600 \
    --camera 0 \
    --altitude 2.5
```

::: info
Use `sudo` because access to `/dev/ttyAMA0` typically requires elevated permissions.
Alternatively, add the user to the `dialout` group with `sudo usermod -aG dialout $USER` and log out/in.
:::

## MAVSDK UDP Bridge Setup

The mission script uses MAVSDK over UDP (port 14540).
Start [mavlink-router](https://github.com/mavlink-router/mavlink-router) in the background to bridge the serial link to UDP:

```sh
sudo apt install -y mavlink-router
mavlink-routerd -e 127.0.0.1:14540 /dev/ttyAMA0:57600 &
```

## Verifying the Vision Estimate

Before the first flight, verify that PX4 is receiving and fusing the vision data:

1. Open *QGroundControl* and connect via a telemetry radio or USB.
2. Open the [MAVLink Inspector](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/analyze_view/mavlink_inspector.html) (**Analyse > MAVLink Inspector**).
3. Confirm that `VISION_POSITION_ESTIMATE` messages are arriving.
4. Move the drone horizontally and verify that the `x` and `y` values change accordingly.

## Safety Considerations

- Always perform initial tests in an open area clear of people and obstacles.
- Set `COM_RCL_EXCEPT` to allow offboard mode without RC override if flying without a radio:
  `COM_RCL_EXCEPT = 4` (bit 2 set).
- Configure a [Return-to-Launch (RTL) failsafe](../config/safety.md) for loss of companion-computer connection.
- Do not fly above visual line of sight (VLOS) during initial testing.

## Further Information

- [Raspberry Pi Companion with Pixhawk](../companion_computer/pixhawk_rpi.md)
- [Visual Inertial Odometry (VIO)](../computer_vision/visual_inertial_odometry.md)
- [Optical Flow](../sensor/optical_flow.md)
- [MAVSDK](../robotics/mavsdk.md)
- [EKF2 Tuning — External Vision System](../advanced_config/tuning_the_ecl_ekf.md#external-vision-system)
