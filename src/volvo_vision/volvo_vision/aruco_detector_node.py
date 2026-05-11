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

        # Tuned params for stable detection under flat/uniform lighting
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

        # CLAHE — better than equalizeHist for uniform backgrounds
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.bridge         = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub_debug    = self.create_publisher(Image, '/volvo/aruco/debug_image', 10)
        self.pub_detected = self.create_publisher(Bool,  '/paper_detected', 10)

        cv2.namedWindow("ArUco Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ArUco Debug", 1280, 720)
        self.get_logger().info("ArUco Node Running — stable detection mode")

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CLAHE on full image
        enhanced = self.clahe.apply(gray)
        # Gentle blur to kill sensor noise without killing marker edges
        blurred  = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return blurred

    def detect_best(self, gray):
        """Try raw gray first, then sharpen pass — return whichever gives more markers."""
        corners1, ids1, _ = self.detector.detectMarkers(gray)
        count1 = len(ids1) if ids1 is not None else 0
        if count1 == 4:
            return corners1, ids1

        # Unsharp mask to recover edges lost in flat lighting
        sharp = cv2.addWeighted(gray, 1.5,
                                cv2.GaussianBlur(gray, (5, 5), 0), -0.5, 0)
        corners2, ids2, _ = self.detector.detectMarkers(sharp)
        count2 = len(ids2) if ids2 is not None else 0

        return (corners2, ids2) if count2 >= count1 else (corners1, ids1)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray  = self.preprocess(frame)
        corners, ids = self.detect_best(gray)

        now            = self.get_clock().now().to_msg()
        marker_positions = {}
        paper_detected   = False

        half = self.marker_size / 2.0
        object_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

        cv2.putText(frame, "ARUCO SYSTEM ACTIVE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        n = len(ids) if ids is not None else 0

        if n > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)

        if n == 4:
            # Assign corner index by image position: TL=0 TR=1 BR=2 BL=3
            detections = []
            for i in range(4):
                cx = float(corners[i][0][:, 0].mean())
                cy = float(corners[i][0][:, 1].mean())
                detections.append((cx, cy, corners[i]))

            detections.sort(key=lambda d: d[1])
            top = sorted(detections[:2], key=lambda d: d[0])
            bot = sorted(detections[2:], key=lambda d: d[0])
            ordered = [top[0], top[1], bot[1], bot[0]]  # TL TR BR BL

            for idx, (cx, cy, corner) in enumerate(ordered):
                image_points = corner[0].astype(np.float32)
                success, rvec, tvec = cv2.solvePnP(
                    object_points, image_points,
                    self.camera_matrix, self.dist_coeffs
                )
                if not success:
                    continue
                marker_positions[idx] = tvec.flatten()
                self.broadcast_tf(rvec, tvec, f"aruco_marker_{idx}", now)
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                  rvec, tvec, 0.03)
                cv2.putText(frame, str(idx), (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        elif 0 < n < 4:
            cv2.putText(frame, f"PARTIAL: {n}/4 markers", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "NO MARKERS DETECTED", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if len(marker_positions) == 4:
            p = [marker_positions[i] for i in range(4)]
            centroid = np.mean(p, axis=0)

            x_axis = p[1] - p[0];  x_axis /= np.linalg.norm(x_axis)
            y_axis = p[3] - p[0];  y_axis /= np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis); z_axis /= np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)

            rot  = np.column_stack((x_axis, y_axis, z_axis))
            quat = R.from_matrix(rot).as_quat()
            self.broadcast_tf_quaternion(quat, centroid.reshape(3, 1), "paper_origin", now)

            paper_detected = True
            all_pts   = np.vstack(corners).reshape(-1, 2)
            center_px = tuple(np.mean(all_pts, axis=0).astype(int))
            cv2.circle(frame, center_px, 12, (0, 0, 255), -1)
            cv2.putText(frame, "PAPER ORIGIN", (center_px[0] + 10, center_px[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        self.pub_detected.publish(Bool(data=paper_detected))
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        cv2.imshow("ArUco Debug", frame)
        cv2.waitKey(1)

    def broadcast_tf(self, rvec, tvec, child, stamp):
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
        self.tf_broadcaster.sendTransform(tf)

    def broadcast_tf_quaternion(self, quat, tvec, child, stamp):
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
        self.tf_broadcaster.sendTransform(tf)


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