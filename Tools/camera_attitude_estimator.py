#!/usr/bin/env python3
# BSD 3-Clause License
#
# Copyright (c) 2024, PX4 Development Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
Camera-Based Attitude Estimator for PX4 Autopilot
===================================================

Runs on a Raspberry Pi 4B companion computer, captures frames from an attached
camera, uses OpenCV optical flow to estimate vehicle angular velocity (visual
gyroscope) and attitude, then streams the result to a PX4 flight controller
over a serial MAVLink connection using ATT_POS_MOCAP messages.

Required Python packages (install on the RPi)::

    pip3 install pymavlink opencv-python-headless numpy

Hardware connections::

    RPi 4B GPIO UART TX (pin 8)  →  Pixhawk TELEM2 RX
    RPi 4B GPIO UART RX (pin 10) →  Pixhawk TELEM2 TX
    Shared GND
    Camera attached to RPi CSI port or USB port

PX4 parameter setup (via QGroundControl)::

    MAV_1_CONFIG  = TELEM2     (enable MAVLink on TELEM2)
    SER_TEL2_BAUD = 921600
    EKF2_EV_CTRL  = 15         (fuse horizontal pos, vertical pos, velocity,
                                 and yaw from external vision)
    EKF2_HGT_REF  = Vision     (use vision as height reference)

Usage::

    python3 camera_attitude_estimator.py [--port /dev/serial0]
                                         [--baud 921600]
                                         [--camera 0]
                                         [--focal-length 600]

Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np
from pymavlink import mavutil

# ─────────────────────────────── defaults ────────────────────────────────────

SERIAL_PORT = "/dev/serial0"  # RPi UART connected to Pixhawk TELEM2
BAUD_RATE = 921600
CAMERA_INDEX = 0  # /dev/video0 for USB; 0 for CSI cameras
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30
FOCAL_LENGTH_PX = 600.0  # Camera focal length in pixels — calibrate first!

# ─────────────────────────────── optical-flow parameters ─────────────────────

# Lucas-Kanade sparse optical-flow settings
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Shi-Tomasi corner-detection settings for seeding new feature points
_FEATURE_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=8,
    blockSize=7,
)

# Minimum number of tracked points before a new detection is triggered
_MIN_TRACKED_POINTS = 50

# ─────────────────────────────── geometry helpers ────────────────────────────


def _rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3×3 rotation matrix to ``(roll, pitch, yaw)`` in radians.

    Uses the ZYX / aerospace convention consistent with PX4's NED frame.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return roll, pitch, yaw


def _euler_to_quaternion(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """Convert ``(roll, pitch, yaw)`` in radians to a ``(w, x, y, z)`` unit
    quaternion using the ZYX convention."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z


# ─────────────────────────────── vision core ─────────────────────────────────


def _estimate_rotation_from_flow(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    dt: float,
    focal_px: float,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Estimate the inter-frame rotation from two sets of tracked 2-D points.

    Uses the five-point essential-matrix algorithm (RANSAC) and decomposes
    the resulting matrix into a rotation.

    Parameters
    ----------
    prev_pts:
        Feature points in the *previous* frame, shape ``(N, 2)``.
    curr_pts:
        Corresponding feature points in the *current* frame, shape ``(N, 2)``.
    dt:
        Time elapsed between the two frames in seconds.
    focal_px:
        Camera focal length in pixels used to build the intrinsics matrix.

    Returns
    -------
    R:
        3×3 rotation matrix (camera frame, current ← previous).
    ang_vel:
        ``(p, q, r)`` body-frame angular velocity in rad/s
        (roll rate, pitch rate, yaw rate).
    """
    if len(prev_pts) < 5 or len(curr_pts) < 5:
        return np.eye(3), (0.0, 0.0, 0.0)

    cx = FRAME_WIDTH / 2.0
    cy = FRAME_HEIGHT / 2.0
    K = np.array(
        [
            [focal_px, 0.0, cx],
            [0.0, focal_px, cy],
            [0.0, 0.0, 1.0],
        ]
    )

    E, mask = cv2.findEssentialMat(
        prev_pts,
        curr_pts,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    if E is None:
        return np.eye(3), (0.0, 0.0, 0.0)

    _, R, _t, _ = cv2.recoverPose(E, prev_pts, curr_pts, K, mask=mask)

    # Angular velocity ≈ incremental angle / elapsed time
    roll_d, pitch_d, yaw_d = _rotation_matrix_to_euler(R)
    ang_vel = (roll_d / dt, pitch_d / dt, yaw_d / dt)

    return R, ang_vel


# ─────────────────────────────── main class ──────────────────────────────────


class CameraAttitudeEstimator:
    """Capture frames, estimate attitude with OpenCV, and stream to PX4.

    The estimator:

    1. Detects Shi-Tomasi corner features in the first frame.
    2. Tracks them across subsequent frames with Lucas-Kanade optical flow.
    3. Solves the five-point essential-matrix problem to recover the
       inter-frame rotation *R*.
    4. Integrates the incremental rotation to maintain an absolute attitude
       estimate ``(roll, pitch, yaw)``.
    5. Transmits the attitude as a ``MAVLink ATT_POS_MOCAP`` message at the
       camera frame rate so that PX4's EKF2 can fuse it as an external-
       vision (EV) measurement.
    """

    def __init__(self, port: str, baud: int, camera_idx: int) -> None:
        print(f"[init] Connecting to PX4 on {port} at {baud} baud …")
        self._mav = mavutil.mavlink_connection(
            port,
            baud=baud,
            source_system=255,
            source_component=0,
        )
        self._mav.wait_heartbeat(timeout=10)
        print(
            f"[init] Heartbeat received from "
            f"system {self._mav.target_system}, "
            f"component {self._mav.target_component}"
        )

        print(f"[init] Opening camera {camera_idx} …")
        self._cap = cv2.VideoCapture(camera_idx)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_idx}")

        # Absolute attitude state (radians, ZYX / NED convention)
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0

        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._prev_time: float = time.monotonic()

    # ── private helpers ───────────────────────────────────────────────────────

    def _send_att_pos_mocap(self) -> None:
        """Transmit the current attitude estimate as ``ATT_POS_MOCAP`` (ID 138).

        Position fields are set to ``NaN`` because only the attitude is
        estimated here; PX4 ignores NaN fields in EKF2 fusion.
        """
        q = _euler_to_quaternion(self._roll, self._pitch, self._yaw)
        usec = int(time.time() * 1e6)
        nan = float("nan")
        self._mav.mav.att_pos_mocap_send(
            usec,
            q,   # [w, x, y, z]
            nan, # x — not estimated
            nan, # y — not estimated
            nan, # z — not estimated
        )

    def _send_heartbeat(self) -> None:
        """Send a MAVLink heartbeat identifying this node as an onboard
        controller so that PX4 keeps the MAVLink link alive."""
        self._mav.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            mavutil.mavlink.MAV_STATE_ACTIVE,
        )

    def _detect_features(self, gray: np.ndarray) -> np.ndarray | None:
        """Detect Shi-Tomasi corner features in *gray* and return them as an
        ``(N, 1, 2)`` float32 array suitable for ``calcOpticalFlowPyrLK``."""
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **_FEATURE_PARAMS)
        return pts

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, focal_px: float = FOCAL_LENGTH_PX) -> None:
        """Main loop — runs until interrupted by Ctrl-C."""
        print("[run] Streaming attitude estimates to PX4 — press Ctrl-C to stop")

        heartbeat_interval = 1.0
        last_heartbeat = time.monotonic()

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    print("[warn] Camera read failed — retrying …")
                    time.sleep(0.01)
                    continue

                now = time.monotonic()
                dt = max(now - self._prev_time, 1e-4)  # guard against zero
                self._prev_time = now

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self._prev_gray is not None and self._prev_pts is not None:
                    # Track existing features into the new frame
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        self._prev_gray, gray, self._prev_pts, None, **_LK_PARAMS
                    )

                    good_prev = self._prev_pts[status.ravel() == 1]
                    good_curr = curr_pts[status.ravel() == 1]

                    if len(good_prev) >= 5:
                        _R, ang_vel = _estimate_rotation_from_flow(
                            good_prev, good_curr, dt, focal_px
                        )

                        # Integrate angular rates to update absolute attitude
                        p, q, r = ang_vel
                        self._roll += p * dt
                        self._pitch += q * dt
                        self._yaw += r * dt

                        self._send_att_pos_mocap()

                        print(
                            f"[att] roll={math.degrees(self._roll):+7.2f}°  "
                            f"pitch={math.degrees(self._pitch):+7.2f}°  "
                            f"yaw={math.degrees(self._yaw):+7.2f}°  "
                            f"ang_vel=({math.degrees(p):+6.1f}, "
                            f"{math.degrees(q):+6.1f}, "
                            f"{math.degrees(r):+6.1f}) °/s"
                        )

                    # Re-detect when too few features survive tracking
                    if len(good_curr) < _MIN_TRACKED_POINTS:
                        self._prev_pts = self._detect_features(gray)
                    else:
                        self._prev_pts = good_curr.reshape(-1, 1, 2)

                else:
                    # Bootstrap: detect the initial feature set
                    self._prev_pts = self._detect_features(gray)

                self._prev_gray = gray

                # Periodic heartbeat to keep the MAVLink link alive
                if now - last_heartbeat >= heartbeat_interval:
                    self._send_heartbeat()
                    last_heartbeat = now

        except KeyboardInterrupt:
            print("\n[run] Stopped by user")
        finally:
            self._cap.release()
            print("[run] Camera released")


# ─────────────────────────────── CLI ─────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stream OpenCV-based attitude estimates to a PX4 flight controller "
            "over a serial MAVLink link."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--port",
        default=SERIAL_PORT,
        help="Serial port connected to the Pixhawk TELEM2 RX/TX pins",
    )
    p.add_argument(
        "--baud",
        type=int,
        default=BAUD_RATE,
        help="Serial baud rate (must match SER_TEL2_BAUD in PX4)",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="OpenCV camera device index (/dev/videoN → N)",
    )
    p.add_argument(
        "--focal-length",
        type=float,
        default=FOCAL_LENGTH_PX,
        dest="focal_length",
        help=(
            "Camera focal length in pixels. "
            "Obtain this from a proper camera calibration for best results."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv
    )

    # Allow the focal length to propagate into the module-level default used
    # inside _estimate_rotation_from_flow.
    global FOCAL_LENGTH_PX
    FOCAL_LENGTH_PX = args.focal_length

    estimator = CameraAttitudeEstimator(args.port, args.baud, args.camera)
    estimator.run(focal_px=args.focal_length)


if __name__ == "__main__":
    main()
