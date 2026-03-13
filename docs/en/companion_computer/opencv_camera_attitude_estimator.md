# Camera-Based Attitude Estimation (OpenCV, Raspberry Pi 4B)

This page explains how to run a Raspberry Pi 4B as a PX4 companion computer that uses an attached camera and *OpenCV* optical flow to estimate vehicle attitude (roll, pitch, yaw) and angular velocity, then streams those estimates to a Pixhawk flight controller over a serial MAVLink connection.

The companion script [`Tools/camera_attitude_estimator.py`](https://github.com/PX4/PX4-Autopilot/blob/main/Tools/camera_attitude_estimator.py) does the following:

1. Detects Shi-Tomasi corner features in the live camera feed.
2. Tracks them across consecutive frames using Lucas-Kanade sparse optical flow.
3. Recovers the inter-frame rotation with the five-point essential-matrix algorithm (RANSAC).
4. Integrates the incremental rotation to maintain an absolute attitude estimate.
5. Transmits the attitude as a `MAVLink ATT_POS_MOCAP` message (ID 138) so that PX4's EKF2 fuses it as an external-vision (EV) measurement.

This is particularly useful for GPS-denied indoor flight where a conventional IMU alone cannot maintain attitude accuracy.

::: warning
Camera-only attitude estimation drifts over time because integration errors accumulate.
For safety-critical applications always retain an onboard IMU and tune `EKF2_EV_CTRL` to fuse both sources.
:::

## Hardware Requirements

- Raspberry Pi 4B (2 GB RAM or more recommended)
- Pixhawk flight controller with a free `TELEM` port (e.g. Pixhawk 6C)
- Camera — either CSI (Raspberry Pi Camera Module v2 or later) or a USB webcam
- 5V power supply for both the RPi and the Pixhawk (separate BEC recommended)

## Wiring

### Serial Connection (RPi UART ↔ Pixhawk TELEM2)

Connect the RPi GPIO UART directly to the Pixhawk `TELEM2` port:

| RPi 4B GPIO Pin        | Pixhawk TELEM2 Pin    |
| ---------------------- | --------------------- |
| TXD (GPIO 14, pin 8)   | UART5_RX (pin 3)      |
| RXD (GPIO 15, pin 10)  | UART5_TX (pin 2)      |
| Ground (pin 6)         | GND (pin 6)           |

::: info
Do **not** connect the Pixhawk VCC (pin 1) to the RPi; power both boards from their own regulators.
:::

See [Raspberry Pi Companion with Pixhawk](../companion_computer/pixhawk_rpi.md) for a step-by-step wiring guide including photos and a pin-diagram.

### Camera

- **CSI camera** — attach the ribbon cable to the CSI port on the RPi and enable the camera interface with `raspi-config` → **Interface Options** → **Camera**.
- **USB webcam** — plug in; it will appear as `/dev/video0` automatically.

## Software Setup

### Raspberry Pi

Follow the Ubuntu 22.04 and serial-port setup described in [Raspberry Pi Companion with Pixhawk](../companion_computer/pixhawk_rpi.md#ubuntu-setup-on-rpi) before continuing.

Install the Python dependencies:

```sh
pip3 install pymavlink opencv-python-headless numpy
```

::: info
Use `opencv-python-headless` (no GUI) on a headless RPi to avoid pulling in X11 dependencies.
If you need the display window for debugging, install `opencv-python` instead.
:::

Clone or copy the estimator script onto the RPi:

```sh
# If you have the PX4-Autopilot repository on the RPi
python3 /path/to/PX4-Autopilot/Tools/camera_attitude_estimator.py

# Or download the script directly
wget https://raw.githubusercontent.com/PX4/PX4-Autopilot/main/Tools/camera_attitude_estimator.py
```

### PX4 Parameter Configuration

Connect the Pixhawk to your laptop via USB and configure the following parameters in *QGroundControl* (**Vehicle Setup** > **Parameters**):

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| `MAV_1_CONFIG` | `TELEM2` | Enable MAVLink on TELEM2 |
| `SER_TEL2_BAUD` | `921600` | Baud rate — must match `--baud` argument |
| `EKF2_EV_CTRL` | `15` | Fuse horizontal position, vertical position, velocity, and yaw from EV |
| `EKF2_HGT_REF` | `Vision` | Use external-vision height as the altitude reference |
| `EKF2_EV_DELAY` | `50` (ms) | Start here and tune empirically — see [Visual Inertial Odometry](../computer_vision/visual_inertial_odometry.md#tuning-EKF2_EV_DELAY) |

::: info
Reboot the flight controller after changing any of these parameters.
:::

#### Camera Focal Length Calibration

The script requires the camera focal length in pixels to compute the essential matrix correctly.
Run a standard OpenCV camera calibration with a checkerboard pattern and note the `fx` value from the calibration output.
Pass it to the script with `--focal-length <fx>`.

A rough starting value for typical webcams is `600` px (640×480 resolution).

## Running the Estimator

On the RPi, start the estimator:

```sh
python3 camera_attitude_estimator.py \
    --port /dev/serial0 \
    --baud 921600 \
    --camera 0 \
    --focal-length 600
```

The terminal will print a live attitude readout:

```
[init] Connecting to PX4 on /dev/serial0 at 921600 baud …
[init] Heartbeat received from system 1, component 1
[init] Opening camera 0 …
[run] Streaming attitude estimates to PX4 — press Ctrl-C to stop
[att] roll=  +0.12°  pitch=  -0.34°  yaw=  +1.20°  ang_vel=(  +0.1,   -0.2,   +0.0) °/s
…
```

Press **Ctrl-C** to stop.

### Command-Line Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--port` | `/dev/serial0` | Serial device connected to Pixhawk TELEM2 |
| `--baud` | `921600` | Baud rate |
| `--camera` | `0` | OpenCV camera index (`/dev/videoN` → `N`) |
| `--focal-length` | `600` | Camera focal length in pixels |

## Verifying the Estimate

Use the *QGroundControl* [MAVLink Inspector](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/analyze_view/mavlink_inspector.html) to confirm that the flight controller is receiving `ATT_POS_MOCAP` messages and that `EKF2` is fusing them.

You can also verify the fusion is active in *QGroundControl* under **Analyze** > **Log Download** after a short test flight by checking that the `EKF2` EV innovation variances are near zero.

For a detailed pre-flight verification checklist, see [Visual Inertial Odometry — Check/Verify VIO Estimate](../computer_vision/visual_inertial_odometry.md#verify_estimate).

## Limitations and Next Steps

- **Attitude drift** — integration of angular velocity introduces drift.
  Fusing camera attitude with the onboard IMU via `EKF2_EV_CTRL` mitigates this.
- **Scale ambiguity** — the essential-matrix decomposition cannot recover metric translation, so this script does not estimate position.
  Add an optical-flow distance sensor or barometer for altitude hold.
- **Dynamic scenes** — fast motion or motion blur degrades feature tracking.
  Mount the camera on a vibration-damped bracket.
- **No GPU acceleration** — for higher frame rates consider enabling the RPi camera ISP pipeline or using a hardware-accelerated OpenCV build.

## Further Reading

- [Raspberry Pi Companion with Pixhawk](../companion_computer/pixhawk_rpi.md)
- [Visual Inertial Odometry (VIO)](../computer_vision/visual_inertial_odometry.md)
- [Using a Companion Computer with Pixhawk Controllers](../companion_computer/pixhawk_companion.md)
- [EKF2 External Vision System Tuning](../advanced_config/tuning_the_ecl_ekf.md#external-vision-system)
