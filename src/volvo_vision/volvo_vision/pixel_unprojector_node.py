#!/usr/bin/env python3
"""
pixel_unprojector_node.py — volvo_arm Phase 3  (brush v3 compatible)
=====================================================================
NaN reload header format (6 floats):
  NaN, NaN, pause_s, lift_u, lift_v, Z_LIFT_MM

Markers published:
  - outline points     (yellow)
  - fill stroke points (blue, per-stroke color cycling)
  - stroke start dots  (green)
  - Z-lift positions   (red X markers)
  - numbered text labels every 10 points
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose, Point
from visualization_msgs.msg import MarkerArray, Marker
from builtin_interfaces.msg import Duration

import tf2_ros
import numpy as np
import math

K = np.array([
    [1723.0,    0.0, 316.0],
    [   0.0, 1711.0, 287.0],
    [   0.0,    0.0,   1.0]
], dtype=np.float64)

K_INV = np.linalg.inv(K)

PEN_Z_CONTACT = 0.002

# Stroke colors (r,g,b) cycling
STROKE_COLORS = [
    (0.0, 0.5, 1.0),
    (1.0, 0.55, 0.0),
    (0.7, 0.0, 0.9),
    (0.0, 0.9, 0.5),
]


class PixelUnprojectorNode(Node):

    def __init__(self):
        super().__init__('pixel_unprojector_node')

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.create_subscription(
            Float32MultiArray,
            '/volvo/shape/pixels',
            self._pixels_cb,
            10
        )

        self.pub_poses   = self.create_publisher(
            PoseArray, '/volvo/shape/waypoints_3d', latched_qos)
        self.pub_markers = self.create_publisher(
            MarkerArray, '/volvo/shape/waypoints_marker', 10)

        self.get_logger().info('PixelUnprojectorNode (brush v3) ready')

    # ── Main callback ──────────────────────────────────────────────────────────

    def _pixels_cb(self, msg: Float32MultiArray):
        flat = list(msg.data)
        n    = len(flat)
        if n < 2:
            self.get_logger().warn('Empty pixel list')
            return

        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                'paper_origin', 'camera_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
        except Exception as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return

        t     = tf_stamped.transform.translation
        q     = tf_stamped.transform.rotation
        t_vec = np.array([t.x, t.y, t.z], dtype=np.float64)
        R_mat = self._quat_to_matrix(q.x, q.y, q.z, q.w)

        # ── Parse flat array — handle 6-float NaN reload markers ──────────────
        # Each element is one of:
        #   ('pt',    u, v)                           — normal waypoint
        #   ('lift',  pause_s, lift_u, lift_v, z_mm)  — brush reload
        segments = []   # list of ('pt',...) or ('lift',...)
        i = 0
        while i < n - 1:
            u = flat[i]
            v = flat[i + 1]
            if math.isnan(u) or math.isnan(v):
                # Reload marker: NaN NaN pause_s lift_u lift_v Z_LIFT_MM
                if i + 5 < n:
                    pause_s = flat[i + 2]
                    lift_u  = flat[i + 3]
                    lift_v  = flat[i + 4]
                    z_mm    = flat[i + 5]
                    segments.append(('lift', pause_s, lift_u, lift_v, z_mm))
                    i += 6
                else:
                    # malformed — skip 2
                    i += 2
            else:
                segments.append(('pt', u, v))
                i += 2

        self.get_logger().info(
            f'Parsed {len(segments)} segments '
            f'({sum(1 for s in segments if s[0]=="pt")} pts, '
            f'{sum(1 for s in segments if s[0]=="lift")} lifts)')

        # ── Unproject all pt segments ──────────────────────────────────────────
        # Build output list preserving lift markers
        # Each entry: ('pt3d', x,y,z) or ('lift', pause_s, lx,ly, z_lift_m)
        out = []
        fail_count = 0

        for seg in segments:
            if seg[0] == 'lift':
                _, pause_s, lift_u, lift_v, z_mm = seg
                lift_pt = self._unproject(lift_u, lift_v, R_mat, t_vec)
                if lift_pt is not None:
                    out.append(('lift', pause_s,
                                lift_pt[0], lift_pt[1],
                                z_mm / 1000.0))   # mm → m
                else:
                    out.append(('lift', pause_s, 0.0, 0.0, z_mm / 1000.0))
            else:
                _, u, v = seg
                pt = self._unproject(u, v, R_mat, t_vec)
                if pt is None:
                    fail_count += 1
                else:
                    out.append(('pt3d', pt[0], pt[1], PEN_Z_CONTACT))

        valid = sum(1 for o in out if o[0] == 'pt3d')
        lifts = sum(1 for o in out if o[0] == 'lift')
        self.get_logger().info(
            f'Unprojected {valid} pts | {fail_count} ray misses | {lifts} lifts')

        # ── Build PoseArray ────────────────────────────────────────────────────
        pose_array = PoseArray()
        pose_array.header.frame_id = 'paper_origin'
        pose_array.header.stamp    = self.get_clock().now().to_msg()

        for o in out:
            pose = Pose()
            if o[0] == 'pt3d':
                pose.position.x = o[1]
                pose.position.y = o[2]
                pose.position.z = o[3]
            else:  # lift — encode as NaN pose (executor reads this)
                pose.position.x = float('nan')
                pose.position.y = float('nan')
                pose.position.z = float('nan')
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.pub_poses.publish(pose_array)
        self.pub_markers.publish(self._build_markers(out))

        self.get_logger().info(
            f'Published {len(pose_array.poses)} poses (latched)')

    # ── Marker builder ─────────────────────────────────────────────────────────

    def _build_markers(self, out: list) -> MarkerArray:
        ma = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        def base_marker(ns, mid, mtype, frame='paper_origin'):
            m                 = Marker()
            m.header.frame_id = frame
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns              = ns
            m.id              = mid
            m.type            = mtype
            m.action          = Marker.ADD
            m.lifetime        = Duration(sec=0)
            m.color.a         = 1.0
            return m

        # ── Outline points (yellow) ────────────────────────────────────────────
        m_outline = base_marker('outline', 0, Marker.POINTS)
        m_outline.scale.x = 0.004
        m_outline.scale.y = 0.004
        m_outline.color.r = 1.0
        m_outline.color.g = 1.0
        m_outline.color.b = 0.0

        # ── Stroke points — one LINE_STRIP per stroke ──────────────────────────
        stroke_markers   = []   # list of Marker (LINE_STRIP)
        stroke_start_m   = base_marker('stroke_starts', 100, Marker.POINTS)
        stroke_start_m.scale.x = 0.008
        stroke_start_m.scale.y = 0.008
        stroke_start_m.color.r = 0.0
        stroke_start_m.color.g = 1.0
        stroke_start_m.color.b = 0.0

        # ── Z-lift markers (red X) ─────────────────────────────────────────────
        lift_markers = []

        # ── Text labels every 10 points ───────────────────────────────────────
        text_markers = []

        in_outline    = True
        stroke_idx    = 0
        current_strip = None
        pt_count      = 0
        lift_mid      = 200

        for o in out:
            if o[0] == 'pt3d':
                x, y, z = o[1], o[2], o[3]
                p = Point(x=x, y=y, z=z)

                if in_outline:
                    m_outline.points.append(p)
                else:
                    if current_strip is not None:
                        current_strip.points.append(p)

                # Label every 10 points
                if pt_count % 10 == 0:
                    tm = base_marker('labels', 300 + pt_count, Marker.TEXT_VIEW_FACING)
                    tm.pose.position.x = x
                    tm.pose.position.y = y
                    tm.pose.position.z = z + 0.015
                    tm.scale.z         = 0.008
                    tm.color.r         = 1.0
                    tm.color.g         = 1.0
                    tm.color.b         = 1.0
                    tm.text            = str(pt_count)
                    text_markers.append(tm)

                pt_count += 1

            elif o[0] == 'lift':
                _, pause_s, lx, ly, z_lift = o

                in_outline = False  # first lift = end of outline

                # Finish current strip
                if current_strip is not None:
                    stroke_markers.append(current_strip)

                # New stroke LINE_STRIP
                stroke_idx   += 1
                col           = STROKE_COLORS[(stroke_idx - 1) % len(STROKE_COLORS)]
                current_strip = base_marker('stroke', stroke_idx, Marker.LINE_STRIP)
                current_strip.scale.x = 0.003
                current_strip.color.r = col[0]
                current_strip.color.g = col[1]
                current_strip.color.b = col[2]

                # Green dot at stroke start (lift position = end of prev stroke)
                sp = Point(x=lx, y=ly, z=PEN_Z_CONTACT)
                stroke_start_m.points.append(sp)

                # Red X at lift position (elevated)
                lm = base_marker('lifts', lift_mid, Marker.ARROW)
                lift_mid += 1
                lm.scale.x = 0.003   # shaft diameter
                lm.scale.y = 0.006   # head diameter
                lm.scale.z = 0.005
                lm.color.r = 1.0
                lm.color.g = 0.0
                lm.color.b = 0.0
                # Arrow from contact point upward to lift height
                p_base = Point(x=lx, y=ly, z=PEN_Z_CONTACT)
                p_top  = Point(x=lx, y=ly, z=PEN_Z_CONTACT + z_lift)
                lm.points.append(p_base)
                lm.points.append(p_top)

                # Text label showing pause duration
                lt = base_marker('lift_labels', lift_mid + 100, Marker.TEXT_VIEW_FACING)
                lt.pose.position.x = lx
                lt.pose.position.y = ly
                lt.pose.position.z = PEN_Z_CONTACT + z_lift + 0.01
                lt.scale.z         = 0.009
                lt.color.r         = 1.0
                lt.color.g         = 0.4
                lt.color.b         = 0.0
                lt.text            = f'{pause_s:.1f}s'
                text_markers.append(lt)
                lift_markers.append(lm)

        # Append last strip
        if current_strip is not None and current_strip.points:
            stroke_markers.append(current_strip)

        # ── Assemble MarkerArray ───────────────────────────────────────────────
        ma.markers.append(m_outline)
        ma.markers.append(stroke_start_m)
        for sm in stroke_markers:
            ma.markers.append(sm)
        for lm in lift_markers:
            ma.markers.append(lm)
        for tm in text_markers:
            ma.markers.append(tm)

        return ma

    # ── Geometry ───────────────────────────────────────────────────────────────

    def _unproject(self, u, v, R_mat, t_vec):
        pixel_h = np.array([u, v, 1.0], dtype=np.float64)
        d_cam   = K_INV @ pixel_h
        d_cam  /= np.linalg.norm(d_cam)
        d_paper = R_mat @ d_cam
        o_paper = t_vec

        if abs(d_paper[2]) < 1e-9:
            return None
        t_param = -o_paper[2] / d_paper[2]
        if t_param < 0:
            return None

        return o_paper + t_param * d_paper

    def _quat_to_matrix(self, x, y, z, w) -> np.ndarray:
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z),   2*(y*z-x*w)],
            [2*(x*z-y*w),     2*(y*z+x*w), 1-2*(x*x+y*y)]
        ], dtype=np.float64)


def main(args=None):
    rclpy.init(args=args)
    node = PixelUnprojectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()