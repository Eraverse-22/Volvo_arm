#!/usr/bin/env python3
"""
aruco_tf_node.py
=================
Detects the 4 ArUco markers on the drawing paper and publishes:
  - TF frames:  base_link → paper_origin  (centroid of the 4 corners)
                base_link → paper_top_left
                base_link → paper_top_right
                base_link → paper_bottom_left
                base_link → paper_bottom_right
  - /paper_detected  (std_msgs/Bool)   True when all 4 visible
  - /camera/image_raw  (sensor_msgs/Image)  so shape_detection_node
    does not need its own camera handle

HOW IT WORKS
  H maps: pixel (u,v) → robot base_link (x,y) [metres].
  For each detected marker, its pixel-centre is mapped through H to get
  its (x,y) in base_link. Z is taken from the paper_z parameter.
  The four corner positions are averaged to produce paper_origin.
  A static Z-aligned TF is broadcast from base_link with that origin.

CORNER ASSIGNMENT
  Because all 4 markers have the same ID (Canva printed one marker 4×),
  assignment is done by pixel position (TOP-LEFT / TOP-RIGHT / etc.).
  This is identical to the working aruco_calibrate_final.py logic.

PARAMETERS (ROS 2 params, all optional)
  homography_file   path to ~/.ros/homography.yaml
  paper_z           Z of paper surface in base_link [m]  default=-0.006
  camera_device     device path                          default='/dev/video2'
  camera_width      capture width                        default=640
  camera_height     capture height                       default=480
  publish_rate_hz   TF + image publish rate             default=15.0

TOPICS OUT
  /paper_detected            std_msgs/Bool
  /camera/image_raw          sensor_msgs/Image   (bgr8)
  /aruco_debug_image         sensor_msgs/Image   (bgr8, annotated)

TF OUT
  paper_origin
  paper_top_left
  paper_top_right
  paper_bottom_left
  paper_bottom_right
"""

from matplotlib import transforms

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import tf2_ros
from geometry_msgs.msg import TransformStamped

import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
import os
import math
import time


# ── Constants ──────────────────────────────────────────────────────────────────
ARUCO_DICT      = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
CORNER_NAMES    = ['TOP-LEFT', 'TOP-RIGHT', 'BOTTOM-LEFT', 'BOTTOM-RIGHT']
FRAME_NAMES     = {
    'TOP-LEFT':     'paper_top_left',
    'TOP-RIGHT':    'paper_top_right',
    'BOTTOM-LEFT':  'paper_bottom_left',
    'BOTTOM-RIGHT': 'paper_bottom_right',
}
# Minimum consecutive frames with all 4 markers before /paper_detected goes True
STABLE_FRAMES   = 5


# ── Homography loader (same as calibration scripts) ────────────────────────────
def _load_H(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    raw = data.get('H') or data.get('homography')
    if raw is None:
        raise ValueError(f"No 'H' or 'homography' key in {path}")
    raw_data = raw if isinstance(raw, list) else raw.get('data', raw)
    return np.array(raw_data, dtype=np.float64).reshape(3, 3)


def _pixel_to_robot(H: np.ndarray, u: float, v: float):
    """pixel (u,v) → robot base_link (x,y) via homography."""
    q = H @ np.array([u, v, 1.0])
    return float(q[0] / q[2]), float(q[1] / q[2])


def _image_position(cx: int, cy: int, w: int, h: int) -> str:
    """Return 'TOP-LEFT' / 'TOP-RIGHT' / 'BOTTOM-LEFT' / 'BOTTOM-RIGHT'."""
    v = 'TOP'    if cy < h // 2 else 'BOTTOM'
    u = 'LEFT'   if cx < w // 2 else 'RIGHT'
    return f'{v}-{u}'


def _make_transform(frame_id: str, child_id: str,
                    x: float, y: float, z: float,
                    stamp) -> TransformStamped:
    """Build a purely translational TransformStamped (no rotation)."""
    t                        = TransformStamped()
    t.header.stamp           = stamp
    t.header.frame_id        = frame_id
    t.child_frame_id         = child_id
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z
    # Identity rotation — paper frame is parallel to base_link XY plane
    t.transform.rotation.x  = 0.0
    t.transform.rotation.y  = 0.0
    t.transform.rotation.z  = 0.0
    t.transform.rotation.w  = 1.0
    return t


# ══════════════════════════════════════════════════════════════════════════════
class ArucoTfNode(Node):

    def __init__(self):
        super().__init__('aruco_tf_node')

        # ── Parameters ─────────────────────────────────────────────────────────
        self.declare_parameter('homography_file',
                               os.path.expanduser('~/.ros/homography.yaml'))
        self.declare_parameter('paper_z',         -0.006)
        self.declare_parameter('camera_device',   '/dev/video2')
        self.declare_parameter('camera_width',    640)
        self.declare_parameter('camera_height',   480)
        self.declare_parameter('publish_rate_hz', 15.0)

        h_file       = self.get_parameter('homography_file').value
        self.paper_z = self.get_parameter('paper_z').value
        cam_dev      = self.get_parameter('camera_device').value
        cam_w        = self.get_parameter('camera_width').value
        cam_h        = self.get_parameter('camera_height').value
        rate_hz      = self.get_parameter('publish_rate_hz').value

        # ── Load homography ─────────────────────────────────────────────────────
        self.H = _load_H(h_file)
        self.get_logger().info(f'Homography loaded from {h_file}')

        # ── Open camera ─────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(cam_dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().fatal(f'Cannot open camera: {cam_dev}')
            raise RuntimeError(f'Camera open failed: {cam_dev}')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        # Warm-up: discard first 10 frames (exposure settling)
        for _ in range(10):
            self.cap.read()
        self.get_logger().info(f'Camera opened: {cam_dev} @ {cam_w}x{cam_h}')

        # ── ArUco detector ──────────────────────────────────────────────────────
        params           = aruco.DetectorParameters()
        self.detector    = aruco.ArucoDetector(ARUCO_DICT, params)
        self.frame_w     = cam_w
        self.frame_h     = cam_h

        # ── TF broadcaster ──────────────────────────────────────────────────────
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Publishers ──────────────────────────────────────────────────────────
        self.bridge          = CvBridge()
        self.pub_detected    = self.create_publisher(Bool,  '/paper_detected',    10)
        self.pub_image       = self.create_publisher(Image, '/camera/image_raw',  10)
        self.pub_debug       = self.create_publisher(Image, '/aruco_debug_image', 10)

        # ── State ───────────────────────────────────────────────────────────────
        self._stable_count   = 0
        self._paper_detected = False
        # Last known corner positions (base_link x,y) keyed by corner name
        self._corner_pos: dict = {}

        # ── Timer ───────────────────────────────────────────────────────────────
        self.create_timer(1.0 / rate_hz, self._timer_callback)
        self.get_logger().info(
            f'ArucoTfNode ready | rate={rate_hz} Hz | paper_z={self.paper_z} m')

    # ═══════════════════════════════════════════════════════════════════════════
    def _timer_callback(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn('Camera read failed', throttle_duration_sec=5.0)
            return

        now = self.get_clock().now().to_msg()

        # Publish raw image for shape_detection_node
        self.pub_image.publish(
            self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))

        # Detect markers
        corners_list, ids, _ = self.detector.detectMarkers(frame)
        debug = frame.copy()

        if ids is None or len(ids) < 4:
            n = 0 if ids is None else len(ids)
            self._stable_count   = 0
            self._paper_detected = False
            self._publish_detected(False)
            cv2.putText(debug, f'Searching... ({n}/4 markers)',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 100, 255), 2)
            self.pub_debug.publish(
                self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
            return

        # Assign corners by pixel position
        detected: dict = {}
        for i, mid in enumerate(ids.flatten()):
            cx = int(corners_list[i][0][:, 0].mean())
            cy = int(corners_list[i][0][:, 1].mean())
            pos = _image_position(cx, cy, self.frame_w, self.frame_h)
            if pos not in detected:   # first marker in each quadrant wins
                detected[pos] = (cx, cy)

        if len(detected) < 4:
            # Not all 4 quadrants covered
            self._stable_count   = 0
            self._paper_detected = False
            self._publish_detected(False)
            cv2.putText(debug,
                        f'Missing corners: {set(CORNER_NAMES)-set(detected.keys())}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 150, 255), 2)
            aruco.drawDetectedMarkers(debug, corners_list, ids)
            self.pub_debug.publish(
                self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
            return

        # Map pixel centres to base_link coordinates
        corner_robot: dict = {}
        for name, (px, py) in detected.items():
            rx, ry = _pixel_to_robot(self.H, px, py)
            corner_robot[name] = (rx, ry)

        # Stability gate
        self._stable_count += 1
        if self._stable_count >= STABLE_FRAMES:
            self._paper_detected = True
            self._corner_pos     = corner_robot

        # ── Broadcast TF frames ─────────────────────────────────────────────────
        transforms = []

        # paper_origin = centroid of 4 corners
        ox = sum(v[0] for v in corner_robot.values()) / 4.0
        oy = sum(v[1] for v in corner_robot.values()) / 4.0
        transforms.append(
            _make_transform('base_link', 'paper_origin',
                            ox, oy, self.paper_z, now))

        # Individual corner frames
        for name, (rx, ry) in corner_robot.items():
            child = FRAME_NAMES[name]

            # relative to paper origin
            local_x = rx - ox
            local_y = ry - oy

            transforms.append(
                _make_transform(
                    'paper_origin',
                    child,
                    local_x,
                    local_y,
                    0.0,
                    now
                )
            )

        self.tf_broadcaster.sendTransform(transforms)

        # ── Publish /paper_detected ─────────────────────────────────────────────
        self._publish_detected(self._paper_detected)

        # ── Debug image ─────────────────────────────────────────────────────────
        aruco.drawDetectedMarkers(debug, corners_list, ids)
        colors_map = {
            'TOP-LEFT':     (255, 180, 0),
            'TOP-RIGHT':    (0,   180, 255),
            'BOTTOM-LEFT':  (0,   255, 120),
            'BOTTOM-RIGHT': (180, 0,   255),
        }
        for name, (px, py) in detected.items():
            col = colors_map.get(name, (200, 200, 200))
            rx, ry = corner_robot[name]
            cv2.circle(debug, (px, py), 10, col, -1)
            cv2.putText(debug,
                        f'{name[:2]}{name[-1]} ({rx*100:.1f},{ry*100:.1f})cm',
                        (px + 8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        status_col = (0, 255, 0) if self._paper_detected else (0, 200, 255)
        status_txt = (f'PAPER LOCKED | origin=({ox*100:.1f},{oy*100:.1f}) cm'
                      if self._paper_detected
                      else f'Stabilising ({self._stable_count}/{STABLE_FRAMES})')
        cv2.putText(debug, status_txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2)
        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

    # ── Helper ──────────────────────────────────────────────────────────────────
    def _publish_detected(self, state: bool):
        msg      = Bool()
        msg.data = state
        self.pub_detected.publish(msg)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = ArucoTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()