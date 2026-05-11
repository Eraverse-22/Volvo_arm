#!/usr/bin/env python3
"""
pixel_unprojector_node.py — volvo_arm Phase 3
==============================================
Converts pixel waypoints → 3D points in paper_origin frame.

PIPELINE per pixel (u, v):
  1. Unproject (u,v) → ray in camera_frame using K^-1
  2. TF lookup: camera_frame → paper_origin
  3. Transform ray origin + direction into paper_origin frame
  4. Intersect ray with paper plane (Z=0 in paper_origin)
  5. Publish 3D point (x, y, 0) in paper_origin frame

TOPICS
  Sub:  /volvo/shape/pixels            std_msgs/Float32MultiArray
  Pub:  /volvo/shape/waypoints_3d      geometry_msgs/PoseArray
        /volvo/shape/waypoints_marker   visualization_msgs/MarkerArray

NaN,NaN in pixel list = pen-lift separator (preserved as NaN in output)
"""

import rclpy
from rclpy.node import Node
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

# Pen z-offset above paper surface (meters)
# 0.0 = touching paper, increase for hover
PEN_Z_CONTACT = 0.002


class PixelUnprojectorNode(Node):

    def __init__(self):
        super().__init__('pixel_unprojector_node')

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Float32MultiArray,
            '/volvo/shape/pixels',
            self._pixels_cb,
            10
        )

        self.pub_poses   = self.create_publisher(
            PoseArray, '/volvo/shape/waypoints_3d', 10)
        self.pub_markers = self.create_publisher(
            MarkerArray, '/volvo/shape/waypoints_marker', 10)

        self.get_logger().info('PixelUnprojectorNode ready | waiting for /volvo/shape/pixels')

    # ── Main callback ──────────────────────────────────────────────────────────

    def _pixels_cb(self, msg: Float32MultiArray):
        flat = msg.data
        if len(flat) < 2:
            self.get_logger().warn('Empty pixel list received')
            return

        # ── TF lookup: camera_frame → paper_origin ─────────────────────────
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                'paper_origin',    # target frame
                'camera_frame',    # source frame
                rclpy.time.Time(), # latest available
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
        except Exception as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return

        # Extract rotation matrix and translation from TF
        t  = tf_stamped.transform.translation
        q  = tf_stamped.transform.rotation
        t_vec = np.array([t.x, t.y, t.z], dtype=np.float64)
        R_mat = self._quat_to_matrix(q.x, q.y, q.z, q.w)

        # ── Parse pixel list — handle NaN separators ───────────────────────
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
        pts_3d = []
        fail_count = 0

        for (u, v) in pts_2d:
            if math.isnan(u):
                pts_3d.append(None)   # pen-lift marker
                continue

            pt = self._unproject(u, v, R_mat, t_vec)
            if pt is None:
                fail_count += 1
                continue
            pts_3d.append(pt)

        valid = sum(1 for p in pts_3d if p is not None)
        self.get_logger().info(
            f'Unprojected {valid}/{len(pts_2d)} pixels | '
            f'{fail_count} ray-plane misses | '
            f'pen-lifts={sum(1 for p in pts_3d if p is None and not math.isnan(0))}')

        # ── Publish PoseArray ───────────────────────────────────────────────
        pose_array = PoseArray()
        pose_array.header.frame_id = 'paper_origin'
        pose_array.header.stamp    = self.get_clock().now().to_msg()

        for pt in pts_3d:
            pose = Pose()
            if pt is None:
                # Encode pen-lift as NaN position
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

        # ── Publish MarkerArray for RViz visualization ──────────────────────
        self.pub_markers.publish(
            self._build_markers(pts_3d))

        self.get_logger().info(
            f'Published {len(pose_array.poses)} poses to /volvo/shape/waypoints_3d')

    # ── Unprojection math ──────────────────────────────────────────────────────

    def _unproject(self, u, v, R_mat, t_vec):
        """
        Unproject pixel (u,v) to 3D point on Z=0 plane of paper_origin.

        Steps:
          1. pixel → normalized camera ray:  d_cam = K^-1 * [u, v, 1]^T
          2. rotate ray to paper_origin:      d_paper = R * d_cam
          3. ray origin in paper_origin:      o_paper = t_vec
          4. intersect with Z=0:
               t_param = -o_paper.z / d_paper.z
               pt = o_paper + t_param * d_paper
        """
        # Step 1 — ray direction in camera frame
        pixel_h = np.array([u, v, 1.0], dtype=np.float64)
        d_cam   = K_INV @ pixel_h
        d_cam  /= np.linalg.norm(d_cam)

        # Step 2 — rotate to paper_origin frame
        d_paper = R_mat @ d_cam

        # Step 3 — ray origin = camera position in paper_origin frame
        o_paper = t_vec

        # Step 4 — intersect with paper plane Z=0
        if abs(d_paper[2]) < 1e-9:
            return None   # ray parallel to paper plane

        t_param = -o_paper[2] / d_paper[2]

        if t_param < 0:
            return None   # intersection behind camera

        pt = o_paper + t_param * d_paper
        return pt   # (x, y, ~0) in paper_origin frame

    # ── RViz marker builder ────────────────────────────────────────────────────

    def _build_markers(self, pts_3d) -> MarkerArray:
        ma = MarkerArray()

        # Delete all previous markers
        delete_all        = Marker()
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        # Outline points — yellow spheres
        outline_marker               = Marker()
        outline_marker.header.frame_id = 'paper_origin'
        outline_marker.header.stamp    = self.get_clock().now().to_msg()
        outline_marker.ns              = 'outline'
        outline_marker.id              = 1
        outline_marker.type            = Marker.POINTS
        outline_marker.action          = Marker.ADD
        outline_marker.scale.x         = 0.003
        outline_marker.scale.y         = 0.003
        outline_marker.color.r         = 1.0
        outline_marker.color.g         = 1.0
        outline_marker.color.b         = 0.0
        outline_marker.color.a         = 1.0
        outline_marker.lifetime        = Duration(sec=0)

        fill_marker               = Marker()
        fill_marker.header.frame_id = 'paper_origin'
        fill_marker.header.stamp    = self.get_clock().now().to_msg()
        fill_marker.ns              = 'fill'
        fill_marker.id              = 2
        fill_marker.type            = Marker.POINTS
        fill_marker.action          = Marker.ADD
        fill_marker.scale.x         = 0.002
        fill_marker.scale.y         = 0.002
        fill_marker.color.r         = 0.0
        fill_marker.color.g         = 0.5
        fill_marker.color.b         = 1.0
        fill_marker.color.a         = 0.8
        fill_marker.lifetime        = Duration(sec=0)

        # First stroke = outline (before first None), rest = fill
        in_outline = True
        for pt in pts_3d:
            if pt is None:
                in_outline = False
                continue
            p = Point()
            p.x = pt[0]
            p.y = pt[1]
            p.z = PEN_Z_CONTACT
            if in_outline:
                outline_marker.points.append(p)
            else:
                fill_marker.points.append(p)

        ma.markers.append(outline_marker)
        ma.markers.append(fill_marker)
        return ma

    # ── Quaternion → rotation matrix ───────────────────────────────────────────

    def _quat_to_matrix(self, x, y, z, w) -> np.ndarray:
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z),  2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)]
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