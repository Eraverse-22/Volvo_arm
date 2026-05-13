#!/usr/bin/env python3
"""
shape_detection_node.py — volvo_arm Phase 2  (brush edition v3)
================================================================
Changes vs v2:
  - Live debug frame shows paper detection area (green boundary) + contour bbox
  - Brush reload encodes Z-lift: NaN, NaN, pause_s, last_u, last_v, Z_LIFT_MM
    Executor must: move to (last_u, last_v, current_z + Z_LIFT_MM), wait pause_s, descend
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

ROW_SPACING_PX       = 22
COL_SPACING_PX       = 6
OUTLINE_SAMPLES      = 80
MIN_SHAPE_AREA       = 1500
APPROX_EPS           = 0.03
CANNY_LOW            = 30
CANNY_HIGH           = 100
LOCK_FRAMES          = 3
BRUSH_RELOAD_PAUSE_S = 1.2
Z_LIFT_MM            = 30.0   # mm to lift above last stroke point during reload

_PAPER_LOWER = np.array([0,   0, 150], dtype=np.uint8)
_PAPER_UPPER = np.array([180, 80, 255], dtype=np.uint8)
_WIN = 'Shape Detection'


class ShapeDetectionNode(Node):

    def __init__(self):
        super().__init__('shape_detection_node')
        self.bridge = CvBridge()

        self._paper_ready  = False
        self._locked       = False
        self._shape_label  = None
        self._latest_frame = None
        self._latest_debug = None
        self._frame_h      = None
        self._frame_w      = None
        self._paper_mask   = None   # kept for live overlay

        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WIN, 800, 600)

        self.create_subscription(Image, '/camera/image_raw', self._image_cb, 10)
        self.create_subscription(Bool,  '/paper_detected',   self._paper_cb, 10)

        self.pub_pixels = self.create_publisher(Float32MultiArray, '/volvo/shape/pixels',     10)
        self.pub_type   = self.create_publisher(String,            '/volvo/shape/type',        10)
        self.pub_debug  = self.create_publisher(Image,             '/volvo/shape/debug_image', 10)

        self.create_service(Trigger, '/detect_shape', self._detect_srv_cb)
        self.create_timer(0.033, self._display_cb)
        self.create_timer(3.0,   self._watchdog_cb)

        self.get_logger().info('ShapeDetectionNode (brush v3) ready — call /detect_shape')

    # ── Axis flip ──────────────────────────────────────────────────────────────

    def _flip_uv(self, pts: list) -> list:
        if self._frame_h is None or self._frame_w is None:
            return pts
        H, W = self._frame_h, self._frame_w
        return [((W - 1) - u, (H - 1) - v) for (u, v) in pts]

    # ── Watchdog ───────────────────────────────────────────────────────────────

    def _watchdog_cb(self):
        self.get_logger().info(
            f'[watchdog] frame={self._latest_frame is not None} | '
            f'paper_ready={self._paper_ready} | locked={self._locked}')

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _paper_cb(self, msg: Bool):
        self._paper_ready = msg.data

    def _image_cb(self, msg: Image):
        self._latest_frame = msg
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._frame_h, self._frame_w = frame.shape[:2]
        debug = frame.copy()

        # ── Always show paper detection area ──────────────────────────────────
        paper_mask = self._detect_paper(frame)
        self._paper_mask = paper_mask
        paper_px = cv2.countNonZero(paper_mask)

        # Draw paper boundary in green
        paper_cnts, _ = cv2.findContours(
            paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug, paper_cnts, -1, (0, 220, 0), 2)

        # Semi-transparent green fill over detected paper region
        overlay = debug.copy()
        cv2.drawContours(overlay, paper_cnts, -1, (0, 80, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.18, debug, 0.82, 0, debug)

        # Show paper pixel count
        cv2.putText(debug, f'paper_px={paper_px}',
                    (10, self._frame_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

        # ── Shape contour live preview ─────────────────────────────────────────
        if not self._locked and paper_px > 5000:
            cnt, name = self._find_shape(frame, paper_mask)
            if cnt is not None:
                cv2.drawContours(debug, [cnt], -1, (0, 255, 255), 2)
                x_bb, y_bb, bw_bb, bh_bb = cv2.boundingRect(cnt)
                cv2.rectangle(debug,
                              (x_bb, y_bb),
                              (x_bb + bw_bb, y_bb + bh_bb),
                              (255, 80, 0), 1)
                cx, cy = self._centroid(cnt)
                area   = cv2.contourArea(cnt)
                cv2.putText(debug, f'{name} | area={int(area)}',
                            (x_bb, y_bb - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                # crosshair at centroid
                cv2.drawMarker(debug, (cx, cy), (255, 255, 0),
                               cv2.MARKER_CROSS, 16, 1)

        # ── Status banner ──────────────────────────────────────────────────────
        if self._locked:
            status_txt   = f'LOCKED | {self._shape_label}'
            status_color = (0, 165, 255)
        elif paper_px > 5000:
            if not self._paper_ready:
                self._paper_ready = True
            status_txt   = 'PAPER OK | call /detect_shape'
            status_color = (0, 255, 0)
        else:
            status_txt   = f'Waiting for paper... ({paper_px} px)'
            status_color = (80, 80, 80)

        cv2.putText(debug, status_txt,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        self._latest_debug = debug

    def _display_cb(self):
        if self._latest_debug is not None:
            cv2.imshow(_WIN, self._latest_debug)
            cv2.waitKey(1)

    # ── Service handler ────────────────────────────────────────────────────────

    def _detect_srv_cb(self, request, response):
        if self._locked:
            response.success = True
            response.message = f'Already locked on: {self._shape_label}'
            return response

        if self._latest_frame is None:
            response.success = False
            response.message = 'No camera frame received yet'
            return response

        if not self._paper_ready:
            frame      = self.bridge.imgmsg_to_cv2(self._latest_frame, desired_encoding='bgr8')
            paper_mask = self._detect_paper(frame)
            if cv2.countNonZero(paper_mask) > 5000:
                self._paper_ready = True
            else:
                response.success = False
                response.message = 'Paper not visible'
                return response

        shape_name   = None
        contour      = None
        stable_count = 0
        last_shape   = None
        deadline     = time.time() + 3.0

        while time.time() < deadline:
            if self._latest_frame is None:
                time.sleep(0.05)
                continue

            frame      = self.bridge.imgmsg_to_cv2(self._latest_frame, desired_encoding='bgr8')
            paper_mask = self._detect_paper(frame)
            cnt, name  = self._find_shape(frame, paper_mask)

            if cnt is None:
                stable_count = 0
                last_shape   = None
                time.sleep(0.05)
                continue

            stable_count = stable_count + 1 if name == last_shape else 1
            last_shape   = name
            self.get_logger().info(f'  stability: {name} {stable_count}/{LOCK_FRAMES}')

            if stable_count >= LOCK_FRAMES:
                shape_name = name
                contour    = cnt
                break

            time.sleep(0.08)

        if contour is None:
            response.success = False
            response.message = 'Shape not stable after 3s'
            return response

        frame        = self.bridge.imgmsg_to_cv2(self._latest_frame, desired_encoding='bgr8')
        outline_pts  = self._sample_outline(contour)
        fill_strokes = self._brush_fill(contour, frame.shape)

        outline_pts_out  = self._flip_uv(outline_pts)
        fill_strokes_out = [self._flip_uv(s) for s in fill_strokes]

        # ── Flat array encoding ────────────────────────────────────────────────
        # Normal point:   [u, v]
        # Reload marker:  [NaN, NaN, pause_s, lift_u, lift_v, Z_LIFT_MM]
        #   lift_u, lift_v = last point of the just-finished stroke (robot stays XY)
        #   executor: move to (lift_u, lift_v, z_current + Z_LIFT_MM), wait, descend
        flat = []
        for (u, v) in outline_pts_out:
            flat += [float(u), float(v)]

        for si, stroke in enumerate(fill_strokes_out):
            if not stroke:
                continue
            last_u, last_v = stroke[-1]          # XY position to lift from
            flat += [_NAN, _NAN,
                     float(BRUSH_RELOAD_PAUSE_S),
                     float(last_u), float(last_v),
                     float(Z_LIFT_MM)]
            for (u, v) in stroke:
                flat += [float(u), float(v)]

        msg_pixels      = Float32MultiArray()
        msg_pixels.data = flat
        self.pub_pixels.publish(msg_pixels)

        msg_type      = String()
        msg_type.data = shape_name
        self.pub_type.publish(msg_type)

        # ── Debug image (original pixel space) ────────────────────────────────
        debug = frame.copy()

        # Paper overlay
        paper_mask_det = self._detect_paper(frame)
        paper_cnts_det, _ = cv2.findContours(
            paper_mask_det, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug, paper_cnts_det, -1, (0, 220, 0), 2)
        overlay2 = debug.copy()
        cv2.drawContours(overlay2, paper_cnts_det, -1, (0, 60, 0), cv2.FILLED)
        cv2.addWeighted(overlay2, 0.15, debug, 0.85, 0, debug)

        # Contour + bbox
        cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)
        x_bb, y_bb, bw_bb, bh_bb = cv2.boundingRect(contour)
        cv2.rectangle(debug,
                      (x_bb, y_bb),
                      (x_bb + bw_bb, y_bb + bh_bb),
                      (255, 80, 0), 1)

        # Fill strokes (original pixel space)
        colors = [(0, 180, 255), (255, 140, 0), (180, 0, 220)]
        for si, stroke in enumerate(fill_strokes):       # original space for display
            if not stroke:
                continue
            col = colors[si % len(colors)]
            for i in range(len(stroke) - 1):
                cv2.line(debug,
                         (int(stroke[i][0]),   int(stroke[i][1])),
                         (int(stroke[i+1][0]), int(stroke[i+1][1])),
                         col, ROW_SPACING_PX // 2)
            # Mark lift point at stroke end
            eu, ev = int(stroke[-1][0]), int(stroke[-1][1])
            cv2.drawMarker(debug, (eu, ev), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.putText(debug,
                        f'+{int(Z_LIFT_MM)}mm / {BRUSH_RELOAD_PAUSE_S}s',
                        (eu + 6, ev - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)

        cx, cy = self._centroid(contour)
        total  = len(outline_pts) + sum(len(s) for s in fill_strokes)
        cv2.putText(debug, shape_name, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(debug,
                    f'{total} pts | {len(fill_strokes)} strokes | LOCKED | flip=UV',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug,
                    f'lift={int(Z_LIFT_MM)}mm x {len(fill_strokes)} | '
                    f'reload={BRUSH_RELOAD_PAUSE_S}s each',
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        self._latest_debug = debug
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

        self._shape_label = shape_name
        self._locked      = True

        self.get_logger().info(
            f'LOCKED: {shape_name} | {total} pts | '
            f'outline={len(outline_pts)} | strokes={len(fill_strokes)} | '
            f'z_lift={Z_LIFT_MM}mm | reload={BRUSH_RELOAD_PAUSE_S}s | flip=BOTH')

        response.success = True
        response.message = (
            f'{shape_name} | {total} pts | {len(fill_strokes)} strokes | '
            f'z_lift={Z_LIFT_MM}mm | reload={BRUSH_RELOAD_PAUSE_S}s each')
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
            d = math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
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

    # ── Brush-aware boustrophedon fill ─────────────────────────────────────────

    def _brush_fill(self, contour, frame_shape) -> list:
        hf, wf = frame_shape[:2]
        mask = np.zeros((hf, wf), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        brush_radius = max(1, ROW_SPACING_PX // 2)
        k_shape = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (brush_radius * 2 + 1, brush_radius * 2 + 1))
        inner = cv2.erode(mask, k_shape, iterations=1)

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