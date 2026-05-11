#!/usr/bin/env python3
"""
shape_detection_node.py — volvo_arm Phase 2
============================================
Service-triggered shape detector.
Publishes pixel waypoints on /volvo/shape/pixels

TOPICS
  Sub:  /camera/image_raw              sensor_msgs/Image
        /paper_detected                std_msgs/Bool  (optional gate)
  Pub:  /volvo/shape/pixels            std_msgs/Float32MultiArray
        /volvo/shape/type              std_msgs/String
        /volvo/shape/debug_image       sensor_msgs/Image

SERVICE
  /detect_shape                        std_srvs/Trigger
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String, Bool
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

import cv2
import numpy as np
import math
import time

_NAN = float('nan')

# ── Tunable ────────────────────────────────────────────────────────────────────
ROW_SPACING_PX  = 12
COL_SPACING_PX  = 3
OUTLINE_SAMPLES = 80
MIN_SHAPE_AREA  = 1500      # lowered — square in image looks small
APPROX_EPS      = 0.03
CANNY_LOW       = 30
CANNY_HIGH      = 100
LOCK_FRAMES     = 3

_PAPER_LOWER = np.array([0,   0, 150], dtype=np.uint8)
_PAPER_UPPER = np.array([180, 80, 255], dtype=np.uint8)


class ShapeDetectionNode(Node):

    def __init__(self):
        super().__init__('shape_detection_node')
        self.bridge = CvBridge()

        # State
        self._paper_ready  = False
        self._locked       = False
        self._shape_label  = None
        self._latest_frame = None
        self._latest_debug = None

        # Subscriptions
        self.create_subscription(
            Image, '/camera/image_raw', self._image_cb, 10)
        self.create_subscription(
            Bool, '/paper_detected', self._paper_cb, 10)

        # Publishers
        self.pub_pixels = self.create_publisher(
            Float32MultiArray, '/volvo/shape/pixels', 10)
        self.pub_type   = self.create_publisher(
            String, '/volvo/shape/type', 10)
        self.pub_debug  = self.create_publisher(
            Image, '/volvo/shape/debug_image', 10)

        # Service
        self.create_service(
            Trigger, '/detect_shape', self._detect_srv_cb)

        # Display timer — safe, runs in executor
        self.create_timer(0.033, self._display_cb)

        # Watchdog timer — print status every 3s
        self.create_timer(3.0, self._watchdog_cb)

        self.get_logger().info(
            'ShapeDetectionNode ready | call /detect_shape to trigger')

    # ── Watchdog ───────────────────────────────────────────────────────────────

    def _watchdog_cb(self):
        frame_ok = self._latest_frame is not None
        self.get_logger().info(
            f'[watchdog] frame={frame_ok} | paper_ready={self._paper_ready} | locked={self._locked}')

    # ── Paper gate ─────────────────────────────────────────────────────────────

    def _paper_cb(self, msg: Bool):
        self._paper_ready = msg.data

    # ── Image buffer ───────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        self._latest_frame = msg

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        debug = frame.copy()

        if self._locked:
            cv2.putText(debug, f'LOCKED | {self._shape_label}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 165, 255), 2)
        elif not self._paper_ready:
            # ── KEY CHANGE: bypass paper gate if frame has white region ──────
            paper_mask = self._detect_paper(frame)
            white_px   = cv2.countNonZero(paper_mask)
            if white_px > 5000:
                self._paper_ready = True
                cv2.putText(debug, 'AUTO-DETECTED PAPER | call /detect_shape',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 200, 0), 2)
            else:
                cv2.putText(debug,
                            f'Waiting paper... white_px={white_px}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (100, 100, 100), 2)
        else:
            cv2.putText(debug, 'READY | call /detect_shape',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        self._latest_debug = debug

    def _display_cb(self):
        if self._latest_debug is not None:
            cv2.imshow('Shape Detection', self._latest_debug)
            cv2.waitKey(1)

    # ── Service handler ────────────────────────────────────────────────────────

    def _detect_srv_cb(self, request, response):
        """
        NO rclpy.spin_once() here — deadlock risk.
        We collect frames by waiting on wall time instead.
        """
        if self._locked:
            response.success = True
            response.message = f'Already locked on: {self._shape_label}'
            return response

        if self._latest_frame is None:
            response.success = False
            response.message = 'No camera frame received yet'
            return response

        # Auto-bypass paper gate if white region is visible
        if not self._paper_ready:
            frame      = self.bridge.imgmsg_to_cv2(
                self._latest_frame, desired_encoding='bgr8')
            paper_mask = self._detect_paper(frame)
            if cv2.countNonZero(paper_mask) > 5000:
                self._paper_ready = True
                self.get_logger().warn(
                    'paper_ready auto-set from service call')
            else:
                response.success = False
                response.message = 'Paper not visible — check lighting/ArUco node'
                return response

        self.get_logger().info(
            f'detect_shape called — collecting {LOCK_FRAMES} stable frames...')

        shape_name = None
        contour    = None
        stable_count = 0
        last_shape   = None

        # Poll latest frame by wall time — no spin_once
        deadline = time.time() + 3.0          # 3 second timeout

        while time.time() < deadline:
            if self._latest_frame is None:
                time.sleep(0.05)
                continue

            frame      = self.bridge.imgmsg_to_cv2(
                self._latest_frame, desired_encoding='bgr8')
            paper_mask = self._detect_paper(frame)
            cnt, name  = self._find_shape(frame, paper_mask)

            if cnt is None:
                stable_count = 0
                last_shape   = None
                time.sleep(0.05)
                continue

            if name == last_shape:
                stable_count += 1
            else:
                stable_count = 1
                last_shape   = name

            self.get_logger().info(
                f'  stability: {name} {stable_count}/{LOCK_FRAMES}')

            if stable_count >= LOCK_FRAMES:
                shape_name = name
                contour    = cnt
                break

            time.sleep(0.08)

        if contour is None:
            response.success = False
            response.message = 'Shape not stable after 3s — check MIN_SHAPE_AREA or lighting'
            return response

        # ── Build waypoints ────────────────────────────────────────────────────
        frame        = self.bridge.imgmsg_to_cv2(
            self._latest_frame, desired_encoding='bgr8')
        outline_pts  = self._sample_outline(contour)
        fill_strokes = self._interior_fill(contour, frame.shape)

        flat = []
        for (u, v) in outline_pts:
            flat += [float(u), float(v)]
        for stroke in fill_strokes:
            flat += [_NAN, _NAN]
            for (u, v) in stroke:
                flat += [float(u), float(v)]

        msg_pixels      = Float32MultiArray()
        msg_pixels.data = flat
        self.pub_pixels.publish(msg_pixels)

        msg_type      = String()
        msg_type.data = shape_name
        self.pub_type.publish(msg_type)

        # ── Debug image ────────────────────────────────────────────────────────
        debug = frame.copy()
        cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)
        for (u, v) in outline_pts[::3]:
            cv2.circle(debug, (int(u), int(v)), 2, (0, 255, 255), -1)
        colors = [(0, 100, 255), (255, 100, 0), (200, 0, 200)]
        for si, stroke in enumerate(fill_strokes):
            for (u, v) in stroke:
                cv2.circle(debug, (int(u), int(v)), 2,
                           colors[si % len(colors)], -1)
        cx, cy = self._centroid(contour)
        total  = len(outline_pts) + sum(len(s) for s in fill_strokes)
        cv2.putText(debug, shape_name, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(debug,
                    f'{total} pts | {1+len(fill_strokes)} strokes | LOCKED',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self._latest_debug = debug
        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

        self._shape_label = shape_name
        self._locked      = True

        self.get_logger().info(
            f'LOCKED: {shape_name} | {total} pts | '
            f'outline={len(outline_pts)} | fill_rows={len(fill_strokes)}')

        response.success = True
        response.message = (f'{shape_name} | {total} pts | '
                            f'{1 + len(fill_strokes)} strokes')
        return response

    # ── Paper mask ─────────────────────────────────────────────────────────────

    def _detect_paper(self, frame: np.ndarray) -> np.ndarray:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _PAPER_LOWER, _PAPER_UPPER)
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        return mask

    # ── Shape detection ────────────────────────────────────────────────────────

    def _find_shape(self, frame: np.ndarray, paper_mask: np.ndarray):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        edges   = cv2.bitwise_and(edges, edges, mask=paper_mask)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best, best_area = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= MIN_SHAPE_AREA and area > best_area:
                best_area = area
                best      = cnt

        if best is None:
            return None, None
        return best, self._classify(best)

    def _classify(self, contour) -> str:
        peri   = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, APPROX_EPS * peri, True)
        v      = len(approx)
        if v == 3:
            return 'triangle'
        if v == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h) if h > 0 else 1.0
            return 'square' if 0.85 <= aspect <= 1.15 else 'rectangle'
        _, r = cv2.minEnclosingCircle(contour)
        if cv2.contourArea(contour) / (math.pi * r * r + 1e-9) > 0.75:
            return 'circle'
        return 'unknown'

    # ── Outline sampling ───────────────────────────────────────────────────────

    def _sample_outline(self, contour) -> list:
        pts = contour.squeeze().astype(float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        n = len(pts)

        arcs = [0.0]
        for i in range(1, n):
            d = math.hypot(pts[i][0] - pts[i-1][0],
                           pts[i][1] - pts[i-1][1])
            arcs.append(arcs[-1] + d)
        total = arcs[-1]
        if total < 1e-6:
            return []

        sampled = []
        for k in range(OUTLINE_SAMPLES):
            target = (k / OUTLINE_SAMPLES) * total
            lo, hi = 0, n - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if arcs[mid] <= target:
                    lo = mid
                else:
                    hi = mid
            seg = max(arcs[hi] - arcs[lo], 1e-9)
            t   = (target - arcs[lo]) / seg
            x   = pts[lo][0] + t * (pts[hi][0] - pts[lo][0])
            y   = pts[lo][1] + t * (pts[hi][1] - pts[lo][1])
            sampled.append((x, y))

        sampled.append(sampled[0])
        return sampled

    # ── Boustrophedon fill ─────────────────────────────────────────────────────

    def _interior_fill(self, contour, frame_shape) -> list:
        hf, wf = frame_shape[:2]
        mask = np.zeros((hf, wf), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        erode_px = max(1, ROW_SPACING_PX // 3)
        k_shape  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        inner    = cv2.erode(mask, k_shape, iterations=1)

        x_bb, y_bb, bw_bb, bh_bb = cv2.boundingRect(contour)

        strokes, l2r = [], True
        for row_y in range(y_bb + ROW_SPACING_PX // 2,
                           y_bb + bh_bb,
                           ROW_SPACING_PX):
            if row_y < 0 or row_y >= hf:
                continue
            col_slice = inner[row_y, max(0, x_bb):min(wf, x_bb + bw_bb)]
            row_xs    = np.where(col_slice == 255)[0] + max(0, x_bb)
            if len(row_xs) < 2:
                continue
            sampled = row_xs[::COL_SPACING_PX]
            if not l2r:
                sampled = sampled[::-1]
            strokes.append([(float(u), float(row_y)) for u in sampled])
            l2r = not l2r

        return strokes

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _centroid(self, contour):
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return 0, 0
        return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()