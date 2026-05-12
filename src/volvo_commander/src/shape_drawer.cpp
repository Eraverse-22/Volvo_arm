/**
 * shape_drawer.cpp — volvo_arm Phase 4
 * =====================================
 * Subscribes to /volvo/shape/waypoints_3d (PoseArray, latched)
 * On /start_drawing (Bool=true): transforms each waypoint
 * paper_origin → base_link, executes on real hardware.
 *
 * NaN poses = pen-lift: raise to Z_HOVER, move XY, lower to Z_TOUCH
 * Normal poses = draw: move at Z_TOUCH
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>

static const std::string PLANNING_GROUP = "volvo_arm";
static const std::string EEF_LINK       = "EE";
static const std::string SOURCE_FRAME   = "paper_origin";
static const std::string TARGET_FRAME   = "base_link";

static constexpr double VEL_DRAW   = 0.15;   // drawing speed
static constexpr double VEL_TRAVEL = 0.3;    // pen-lift travel speed
static constexpr double ACC_SCALE  = 0.1;

static constexpr double Z_TOUCH    = 0.017;  // pen touching paper
static constexpr double Z_HOVER    = 0.134;  // pen lifted (travel height)

// Safety
static constexpr double Z_MIN         = 0.010;
static constexpr double Z_MAX         = 0.200;
static constexpr double XY_RADIUS_MAX = 0.450;

inline bool is_nan_pose(const geometry_msgs::msg::Pose& p)
{
    return std::isnan(p.position.x) ||
           std::isnan(p.position.y) ||
           std::isnan(p.position.z);
}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>(
        "shape_drawer",
        rclcpp::NodeOptions()
            .automatically_declare_parameters_from_overrides(true)
    );

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinner([&executor]() { executor.spin(); });

    auto tf_buffer   = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    auto tf_listener = std::make_shared<tf2_ros::TransformListener>(
        *tf_buffer, node);

    auto arm = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        node, PLANNING_GROUP);

    arm->setEndEffectorLink(EEF_LINK);
    arm->setMaxVelocityScalingFactor(VEL_DRAW);
    arm->setMaxAccelerationScalingFactor(ACC_SCALE);
    arm->setPlanningTime(5.0);

    // ── Shared waypoints ──────────────────────────────────────────────────────
    geometry_msgs::msg::PoseArray::SharedPtr waypoints;
    std::mutex                               wp_mutex;
    std::atomic<bool>                        drawing = false;

    // ── Latched subscriber — matches unprojector QoS ──────────────────────────
    rclcpp::QoS latched_qos(1);
    latched_qos.transient_local().reliable();

    auto sub_wp = node->create_subscription<geometry_msgs::msg::PoseArray>(
        "/volvo/shape/waypoints_3d", latched_qos,
        [&](const geometry_msgs::msg::PoseArray::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(wp_mutex);
            waypoints = msg;
            RCLCPP_INFO(node->get_logger(),
                "Waypoints received: %zu poses", msg->poses.size());
        }
    );

    // ── Helper: move to pose in base_link ─────────────────────────────────────
    auto move_to = [&](double x, double y, double z, double vel) -> bool {
        // Safety check
        double r = std::sqrt(x*x + y*y);
        if (z < Z_MIN || z > Z_MAX || r > XY_RADIUS_MAX) {
            RCLCPP_ERROR(node->get_logger(),
                "SAFETY ABORT: x=%.3f y=%.3f z=%.3f r=%.3f", x, y, z, r);
            return false;
        }

        geometry_msgs::msg::PoseStamped target;
        target.header.frame_id    = TARGET_FRAME;
        target.header.stamp       = node->get_clock()->now();
        target.pose.position.x    = x;
        target.pose.position.y    = y;
        target.pose.position.z    = z;
        target.pose.orientation.x = 1.0;
        target.pose.orientation.y = 0.0;
        target.pose.orientation.z = 0.0;
        target.pose.orientation.w = 0.0;

        arm->setMaxVelocityScalingFactor(vel);
        arm->setStartStateToCurrentState();
        arm->setPoseTarget(target);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        bool ok = (arm->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if (!ok) {
            RCLCPP_WARN(node->get_logger(),
                "Plan failed for x=%.3f y=%.3f z=%.3f", x, y, z);
            return false;
        }
        return (arm->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    };

    // ── /start_drawing subscriber ──────────────────────────────────────────────
    auto sub_start = node->create_subscription<std_msgs::msg::Bool>(
        "/start_drawing", 10,
        [&](const std_msgs::msg::Bool::SharedPtr msg) {
            if (!msg->data)  return;
            if (drawing) {
                RCLCPP_WARN(node->get_logger(), "Already drawing — ignoring.");
                return;
            }

            geometry_msgs::msg::PoseArray::SharedPtr wp_copy;
            {
                std::lock_guard<std::mutex> lock(wp_mutex);
                wp_copy = waypoints;
            }

            if (!wp_copy || wp_copy->poses.empty()) {
                RCLCPP_ERROR(node->get_logger(),
                    "No waypoints received yet — call /detect_shape first.");
                return;
            }

            drawing = true;

            std::thread([&, wp_copy]() {

                RCLCPP_INFO(node->get_logger(),
                    "Starting draw: %zu waypoints", wp_copy->poses.size());

                // ── Pre-transform all waypoints to base_link ───────────────────
                struct WP { double x, y; bool is_lift; };
                std::vector<WP> wps;
                wps.reserve(wp_copy->poses.size());

                for (auto& pose : wp_copy->poses) {
                    if (is_nan_pose(pose)) {
                        wps.push_back({0, 0, true});
                        continue;
                    }

                    geometry_msgs::msg::PoseStamped ps_in, ps_out;
                    ps_in.header.frame_id    = SOURCE_FRAME;
                    ps_in.header.stamp       = node->get_clock()->now();
                    ps_in.pose               = pose;
                    ps_in.pose.position.z    = 0.0;   // z handled separately

                    try {
                        ps_out = tf_buffer->transform(
                            ps_in, TARGET_FRAME,
                            tf2::durationFromSec(1.0));
                    } catch (const tf2::TransformException& ex) {
                        RCLCPP_ERROR(node->get_logger(),
                            "TF failed: %s", ex.what());
                        drawing = false;
                        return;
                    }

                    wps.push_back({
                        ps_out.pose.position.x,
                        ps_out.pose.position.y,
                        false
                    });
                }

                RCLCPP_INFO(node->get_logger(),
                    "All %zu waypoints transformed. Starting execution...",
                    wps.size());

                // ── Step 1: hover to first point ───────────────────────────────
                size_t first_draw = 0;
                while (first_draw < wps.size() && wps[first_draw].is_lift)
                    first_draw++;

                if (first_draw >= wps.size()) {
                    RCLCPP_ERROR(node->get_logger(), "No drawable waypoints.");
                    drawing = false;
                    return;
                }

                RCLCPP_INFO(node->get_logger(),
                    "Hovering to start: x=%.3f y=%.3f",
                    wps[first_draw].x, wps[first_draw].y);

                if (!move_to(wps[first_draw].x, wps[first_draw].y,
                             Z_HOVER, VEL_TRAVEL)) {
                    drawing = false;
                    return;
                }

                // ── Step 2: lower pen ──────────────────────────────────────────
                if (!move_to(wps[first_draw].x, wps[first_draw].y,
                             Z_TOUCH, VEL_DRAW)) {
                    drawing = false;
                    return;
                }

                // ── Step 3: execute waypoints ──────────────────────────────────
                size_t drawn = 0;
                bool   pen_down = true;

                for (size_t i = first_draw + 1; i < wps.size(); ++i) {
                    auto& wp = wps[i];

                    if (wp.is_lift) {
                        // Raise pen
                        if (pen_down && i > 0) {
                            auto& prev = wps[i-1];
                            move_to(prev.x, prev.y, Z_HOVER, VEL_TRAVEL);
                        }
                        pen_down = false;
                        continue;
                    }

                    if (!pen_down) {
                        // Travel to next stroke start at hover height
                        move_to(wp.x, wp.y, Z_HOVER, VEL_TRAVEL);
                        // Lower pen
                        move_to(wp.x, wp.y, Z_TOUCH, VEL_DRAW);
                        pen_down = true;
                        drawn++;
                        continue;
                    }

                    // Normal draw move
                    if (!move_to(wp.x, wp.y, Z_TOUCH, VEL_DRAW)) {
                        RCLCPP_WARN(node->get_logger(),
                            "Skipping waypoint %zu — plan failed", i);
                        continue;
                    }
                    drawn++;

                    if (drawn % 20 == 0) {
                        RCLCPP_INFO(node->get_logger(),
                            "Progress: %zu/%zu waypoints drawn",
                            drawn, wps.size());
                    }
                }

                // ── Step 4: lift pen and go home ───────────────────────────────
                if (!wps.empty() && !wps.back().is_lift) {
                    auto& last = wps.back();
                    move_to(last.x, last.y, Z_HOVER, VEL_TRAVEL);
                }

                arm->setNamedTarget("draw_start");
                moveit::planning_interface::MoveGroupInterface::Plan home_plan;
                if (arm->plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS)
                    arm->execute(home_plan);

                RCLCPP_INFO(node->get_logger(),
                    "Drawing complete! %zu waypoints executed.", drawn);

                drawing = false;

            }).detach();
        }
    );

    RCLCPP_INFO(node->get_logger(),
        "\n"
        "─────────────────────────────────────────\n"
        "  shape_drawer ready\n"
        "  Z_TOUCH=%.3f  Z_HOVER=%.3f\n"
        "  Waiting for /volvo/shape/waypoints_3d\n"
        "  Then: ros2 topic pub /start_drawing std_msgs/msg/Bool \"data: true\" --once\n"
        "─────────────────────────────────────────",
        Z_TOUCH, Z_HOVER
    );

    spinner.join();
    rclcpp::shutdown();
    return 0;
}