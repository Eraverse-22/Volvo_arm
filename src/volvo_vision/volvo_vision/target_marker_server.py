#!/usr/bin/env python3
"""
target_marker_server.py
=======================
Spawns a 6-DOF InteractiveMarker in RViz (paper_origin frame).
Drag it to any pose, then:
  ros2 topic pub /run_to_target std_msgs/Bool "data: true" --once

Publishes current marker pose to /arm_test_target (PoseStamped).
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Vector3
from visualization_msgs.msg import (
    InteractiveMarker, InteractiveMarkerControl, Marker)
from interactive_markers.interactive_marker_server import InteractiveMarkerServer


SOURCE_FRAME = 'paper_origin'


class TargetMarkerServer(Node):

    def __init__(self):
        super().__init__('target_marker_server')

        self._pose = Pose()
        self._pose.position.x    = 0.0
        self._pose.position.y    = 0.0
        self._pose.position.z    = 0.0   # 11.7 cm above paper — safe default
        self._pose.orientation.w = 1.0

        self._server = InteractiveMarkerServer(self, 'arm_test_marker')
        self._setup_marker()
        self._server.applyChanges()

        self._pub = self.create_publisher(
            PoseStamped, '/arm_test_target', 10)

        # Publish current pose at 10 Hz so the C++ node always has latest
        self.create_timer(0.1, self._publish_pose)

        self.get_logger().info(
            'TargetMarkerServer ready.\n'
            '  1. Add InteractiveMarkers in RViz → topic: /arm_test_marker/update\n'
            '  2. Drag the green sphere to your target\n'
            '  3. ros2 topic pub /run_to_target std_msgs/msg/Bool '
            '"data: true" --once'
        )

    def _setup_marker(self):
        im = InteractiveMarker()
        im.header.frame_id = SOURCE_FRAME
        im.name            = 'target_pose'
        im.description     = 'EE target — drag me'
        im.scale           = 0.12
        im.pose            = self._pose

        # Visual sphere
        sphere          = Marker()
        sphere.type     = Marker.SPHERE
        sphere.scale    = Vector3(x=0.035, y=0.035, z=0.035)
        sphere.color.r  = 0.1
        sphere.color.g  = 1.0
        sphere.color.b  = 0.3
        sphere.color.a  = 0.9

        vis_ctrl                = InteractiveMarkerControl()
        vis_ctrl.always_visible = True
        vis_ctrl.markers.append(sphere)
        im.controls.append(vis_ctrl)

        # 6-DOF controls
        dofs = [
            ('move_x',    InteractiveMarkerControl.MOVE_AXIS,   0.0,   0.0,   0.0,   1.0),
            ('move_y',    InteractiveMarkerControl.MOVE_AXIS,   0.0,   0.0,   0.707, 0.707),
            ('move_z',    InteractiveMarkerControl.MOVE_AXIS,   0.0,  -0.707, 0.0,   0.707),
            ('rotate_x',  InteractiveMarkerControl.ROTATE_AXIS, 1.0,   0.0,   0.0,   0.0),
            ('rotate_y',  InteractiveMarkerControl.ROTATE_AXIS, 0.0,   1.0,   0.0,   0.0),
            ('rotate_z',  InteractiveMarkerControl.ROTATE_AXIS, 0.0,   0.0,   1.0,   0.0),
        ]
        for name, mode, ox, oy, oz, ow in dofs:
            ctrl                  = InteractiveMarkerControl()
            ctrl.name             = name
            ctrl.interaction_mode = mode
            ctrl.orientation.x    = ox
            ctrl.orientation.y    = oy
            ctrl.orientation.z    = oz
            ctrl.orientation.w    = ow
            im.controls.append(ctrl)

        self._server.insert(im, feedback_callback=self._feedback_cb)

    def _feedback_cb(self, feedback):
        self._pose = feedback.pose

    def _publish_pose(self):
        msg = PoseStamped()
        msg.header.frame_id = SOURCE_FRAME
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.pose            = self._pose
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetMarkerServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()