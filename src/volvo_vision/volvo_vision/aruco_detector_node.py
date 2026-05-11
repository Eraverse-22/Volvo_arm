#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import tf2_ros
import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R


class VolvoArucoDetector(Node):

    def __init__(self):
        super().__init__('aruco_detector_node')

        self.declare_parameter("marker_size", 0.038)
        self.marker_size = self.get_parameter("marker_size").value

        self.camera_matrix = np.array([
            [1723.0,    0.0, 316.0],
            [   0.0, 1711.0, 287.0],
            [   0.0,    0.0,   1.0]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1))

        self.aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

        params = aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin  = 3
        params.adaptiveThreshWinSizeMax  = 53
        params.adaptiveThreshWinSizeStep = 4
        params.adaptiveThreshConstant    = 7
        params.minMarkerPerimeterRate    = 0.03
        params.maxMarkerPerimeterRate    = 0.5
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate     = 0.05
        params.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize   = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy  = 0.1

        self.detector = aruco.ArucoDetector(self.aruco_dict, params)
        self.clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.bridge         = CvBridge()
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # ── Lock state ────────────────────────────────────────────────────────
        self._locked          = False   # True after first good detection
        self._stable_count    = 0
        self._last_positions  = None
        REQUIRED_STABLE       = 5       # consecutive matching frames before lock
        self.REQUIRED_STABLE  = REQUIRED_STABLE
        self._locked_marker_tfs = []    # store for periodic republish

        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub_debug    = self.create_publisher(Image, '/volvo/aruco/debug_image', 10)
        self.pub_detected = self.create_publisher(Bool,  '/paper_detected', 10)

        # Republish static TFs at 1 Hz so RViz and TF tree never time out
        self.create_timer(1.0, self._republish_static)

        cv2.namedWindow("ArUco Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ArUco Debug", 1280, 720)
        self.get_logger().info("ArUco Node — one-shot lock mode (need 5 stable frames)")

    # ── Static TF republisher ─────────────────────────────────────────────────

    def _republish_static(self):
        if self._locked and self._locked_marker_tfs:
            now = self.get_clock().now().to_msg()
            for tf in self._locked_marker_tfs:
                tf.header.stamp = now
            self.tf_broadcaster.sendTransform(self._locked_marker_tfs)

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, frame):
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        return cv2.GaussianBlur(enhanced, (3, 3), 0)

    def detect_best(self, gray):
        corners1, ids1, _ = self.detector.detectMarkers(gray)
        count1 = len(ids1) if ids1 is not None else 0
        if count1 == 4:
            return corners1, ids1
        sharp = cv2.addWeighted(gray, 1.5,
                                cv2.GaussianBlur(gray, (5, 5), 0), -0.5, 0)
        corners2, ids2, _ = self.detector.detectMarkers(sharp)
        count2 = len(ids2) if ids2 is not None else 0
        return (corners2, ids2) if count2 >= count1 else (corners1, ids1)

    # ── Main callback ─────────────────────────────────────────────────────────

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # ── Already locked — just show frozen overlay ─────────────────────────
        if self._locked:
            cv2.putText(frame, "PAPER LOCKED — TF frozen", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, "Restart node to re-detect", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            self.pub_detected.publish(Bool(data=True))
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            cv2.imshow("ArUco Debug", frame)
            cv2.waitKey(1)
            return

        # ── Still searching ───────────────────────────────────────────────────
        gray    = self.preprocess(frame)
        corners, ids = self.detect_best(gray)
        n = len(ids) if ids is not None else 0

        cv2.putText(frame, f"Searching... stable {self._stable_count}/{self.REQUIRED_STABLE}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        half = self.marker_size / 2.0
        object_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

        if n > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)

        if n == 4:
            # Sort into TL=0 TR=1 BR=2 BL=3
            detections = []
            for i in range(4):
                cx = float(corners[i][0][:, 0].mean())
                cy = float(corners[i][0][:, 1].mean())
                detections.append((cx, cy, corners[i]))

            detections.sort(key=lambda d: d[1])
            top = sorted(detections[:2], key=lambda d: d[0])
            bot = sorted(detections[2:], key=lambda d: d[0])
            ordered = [top[0], top[1], bot[1], bot[0]]

            marker_positions = {}
            pose_data = []   # (rvec, tvec, idx)

            for idx, (cx, cy, corner) in enumerate(ordered):
                image_points = corner[0].astype(np.float32)
                success, rvec, tvec = cv2.solvePnP(
                    object_points, image_points,
                    self.camera_matrix, self.dist_coeffs
                )
                if not success:
                    continue
                marker_positions[idx] = tvec.flatten()
                pose_data.append((rvec, tvec, idx, cx, cy))
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                  rvec, tvec, 0.03)
                cv2.putText(frame, str(idx), (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if len(marker_positions) == 4:
                positions_flat = np.array([marker_positions[i] for i in range(4)])

                # Stability gate: check if positions are consistent with last frame
                if self._last_positions is not None:
                    diff = np.max(np.linalg.norm(positions_flat - self._last_positions, axis=1))
                    if diff < 0.005:   # 5mm threshold
                        self._stable_count += 1
                    else:
                        self._stable_count = 0
                else:
                    self._stable_count = 1

                self._last_positions = positions_flat

                cv2.putText(frame,
                            f"Stable: {self._stable_count}/{self.REQUIRED_STABLE}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

                # ── Lock when stable enough ───────────────────────────────────
                if self._stable_count >= self.REQUIRED_STABLE:
                    now = self.get_clock().now().to_msg()
                    tfs = []

                    for rvec, tvec, idx, _, _ in pose_data:
                        tfs.append(self._make_tf(rvec, tvec,
                                                 f"aruco_marker_{idx}", now))

                    # paper_origin
                    p = [marker_positions[i] for i in range(4)]
                    centroid = np.mean(p, axis=0)
                    quat = np.array([0.0, 0.0, 0.0, 1.0])  # No rotation relative to camera frame
                    tfs.append(self._make_tf_quat(quat, centroid.reshape(3,1),
                                                  "paper_origin", now))

                    self.tf_broadcaster.sendTransform(tfs)
                    self._locked_marker_tfs = tfs
                    self._locked = True

                    self.get_logger().info(
                        f"LOCKED — paper_origin at "
                        f"x={centroid[0]:.3f} y={centroid[1]:.3f} z={centroid[2]:.3f} "
                        f"(camera_frame)")

        elif 0 < n < 4:
            self._stable_count = 0
            self._last_positions = None
            cv2.putText(frame, f"PARTIAL: {n}/4", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        else:
            self._stable_count = 0
            self._last_positions = None
            cv2.putText(frame, "NO MARKERS", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        self.pub_detected.publish(Bool(data=False))
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        cv2.imshow("ArUco Debug", frame)
        cv2.waitKey(1)

    # ── TF builders ───────────────────────────────────────────────────────────

    def _make_tf(self, rvec, tvec, child, stamp):
        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = "camera_frame"
        tf.child_frame_id  = child
        tf.transform.translation.x = float(tvec[0])
        tf.transform.translation.y = float(tvec[1])
        tf.transform.translation.z = float(tvec[2])
        rot  = cv2.Rodrigues(rvec)[0]
        quat = R.from_matrix(rot).as_quat()
        tf.transform.rotation.x = float(quat[0])
        tf.transform.rotation.y = float(quat[1])
        tf.transform.rotation.z = float(quat[2])
        tf.transform.rotation.w = float(quat[3])
        return tf

    def _make_tf_quat(self, quat, tvec, child, stamp):
        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = "camera_frame"
        tf.child_frame_id  = child
        tf.transform.translation.x = float(tvec[0])
        tf.transform.translation.y = float(tvec[1])
        tf.transform.translation.z = float(tvec[2])
        tf.transform.rotation.x = float(quat[0])
        tf.transform.rotation.y = float(quat[1])
        tf.transform.rotation.z = float(quat[2])
        tf.transform.rotation.w = float(quat[3])
        return tf


def main(args=None):
    rclpy.init(args=args)
    node = VolvoArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()