#!/usr/bin/env python3
"""
pixel_unprojector_node.py — volvo_arm Phase 3
==============================================
Converts pixel waypoints → 3D points in paper_origin frame.
Uses TRANSIENT_LOCAL (latched) publisher so shape_drawer
can receive waypoints anytime after detection.
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


class PixelUnprojectorNode(Node):

    def __init__(self):
        super().__init__('pixel_unprojector_node')

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Latched QoS — shape_drawer receives even after this node publishes ──
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

        self.get_logger().info(
            'PixelUnprojectorNode ready | waiting for /volvo/shape/pixels')

    def _pixels_cb(self, msg: Float32MultiArray):
        flat = msg.data
        if len(flat) < 2:
            self.get_logger().warn('Empty pixel list received')
            return

        # ── TF lookup: camera_frame → paper_origin ─────────────────────────
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                'paper_origin',
                'camera_frame',
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

        # ── Parse pixel list ────────────────────────────────────────────────
        pts_2d = []
        i = 0
        while i < len(flat) - 1:
            u = flat[i]
            v = flat[i + 1]
            i += 2
            if math.isnan(u) or math.isnan(v):
                pts_2d.append((float('nan'), float('nan')))
            else:
                pts_2d.append((u, v))

        # ── Unproject each pixel ────────────────────────────────────────────
        pts_3d     = []
        fail_count = 0

        for (u, v) in pts_2d:
            if math.isnan(u):
                pts_3d.append(None)
                continue
            pt = self._unproject(u, v, R_mat, t_vec)
            if pt is None:
                fail_count += 1
                continue
            pts_3d.append(pt)

        valid = sum(1 for p in pts_3d if p is not None)
        lifts = sum(1 for p in pts_3d if p is None)
        self.get_logger().info(
            f'Unprojected {valid}/{len(pts_2d)} pixels | '
            f'{fail_count} ray-plane misses | pen-lifts={lifts}')

        # ── Publish PoseArray (latched) ─────────────────────────────────────
        pose_array            = PoseArray()
        pose_array.header.frame_id = 'paper_origin'
        pose_array.header.stamp    = self.get_clock().now().to_msg()

        for pt in pts_3d:
            pose = Pose()
            if pt is None:
                pose.position.x = float('nan')
                pose.position.y = float('nan')
                pose.position.z = float('nan')
            else:
                pose.position.x = pt[0]
                pose.position.y = pt[1]
                pose.position.z = PEN_Z_CONTACT
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.pub_poses.publish(pose_array)
        self.pub_markers.publish(self._build_markers(pts_3d))

        self.get_logger().info(
            f'Published {len(pose_array.poses)} poses to '
            f'/volvo/shape/waypoints_3d (latched)')

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

    def _build_markers(self, pts_3d) -> MarkerArray:
        ma         = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        def make_marker(ns, mid, r, g, b, size):
            m                    = Marker()
            m.header.frame_id    = 'paper_origin'
            m.header.stamp       = self.get_clock().now().to_msg()
            m.ns                 = ns
            m.id                 = mid
            m.type               = Marker.POINTS
            m.action             = Marker.ADD
            m.scale.x            = size
            m.scale.y            = size
            m.color.r            = r
            m.color.g            = g
            m.color.b            = b
            m.color.a            = 1.0
            m.lifetime           = Duration(sec=0)
            return m

        outline = make_marker('outline', 1, 1.0, 1.0, 0.0, 0.003)
        fill    = make_marker('fill',    2, 0.0, 0.5, 1.0, 0.002)

        in_outline = True
        for pt in pts_3d:
            if pt is None:
                in_outline = False
                continue
            p   = Point()
            p.x = pt[0]
            p.y = pt[1]
            p.z = PEN_Z_CONTACT
            if in_outline:
                outline.points.append(p)
            else:
                fill.points.append(p)

        ma.markers.append(outline)
        ma.markers.append(fill)
        return ma

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