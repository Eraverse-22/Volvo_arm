#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <thread>
#include <mutex>
#include <atomic>

static const std::string PLANNING_GROUP = "volvo_arm";
static const std::string EEF_LINK       = "EE";
static const std::string SOURCE_FRAME   = "paper_origin";
static const std::string TARGET_FRAME   = "base_link";
static constexpr double  VEL_SCALE      = 0.1;
static constexpr double  ACC_SCALE      = 0.1;

// ── Safety limits — arm cannot go below or outside these ──────────────────────
static constexpr double  Z_MIN          =  0.010;   // 1cm above table — hard floor
static constexpr double  Z_MAX          =  0.200;   // 20cm above table — hard ceiling
static constexpr double  XY_RADIUS_MAX  =  0.450;   // 45cm from base_link origin

static constexpr double  Z_HOVER  = 0.12;   // draw_start hover height
static constexpr double  Z_TOUCH  = 0.017;   // pen touching paper

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>(
        "arm_target_tester",
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
    arm->setMaxVelocityScalingFactor(VEL_SCALE);
    arm->setMaxAccelerationScalingFactor(ACC_SCALE);
    arm->setPlanningTime(5.0);

    RCLCPP_INFO(node->get_logger(),
        "MoveGroupInterface ready | group=%s | eef=%s | vel=%.2f | acc=%.2f",
        PLANNING_GROUP.c_str(), EEF_LINK.c_str(), VEL_SCALE, ACC_SCALE);

    geometry_msgs::msg::PoseStamped latest_target;
    std::mutex                      target_mutex;
    bool                            has_target = false;
    std::atomic<bool>               executing  = false;

    auto sub_target = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/arm_test_target", 10,
        [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(target_mutex);
            latest_target = *msg;
            has_target    = true;
        }
    );

    auto sub_run = node->create_subscription<std_msgs::msg::Bool>(
        "/run_to_target", 10,
        [&](const std_msgs::msg::Bool::SharedPtr msg) {
            if (!msg->data)  return;
            if (executing)   { RCLCPP_WARN(node->get_logger(), "Already executing."); return; }
            if (!has_target) { RCLCPP_WARN(node->get_logger(), "No target received yet."); return; }

            executing = true;

            geometry_msgs::msg::PoseStamped target_paper;
            {
                std::lock_guard<std::mutex> lock(target_mutex);
                target_paper = latest_target;
            }

            std::thread([&, target_paper]() {

                // Step 1 — TF transform paper_origin → base_link
                geometry_msgs::msg::PoseStamped target_base;
                try {
                    target_base = tf_buffer->transform(
                        target_paper,
                        TARGET_FRAME,
                        tf2::durationFromSec(2.0)
                    );
                } catch (const tf2::TransformException& ex) {
                    RCLCPP_ERROR(node->get_logger(), "TF failed: %s", ex.what());
                    executing = false;
                    return;
                }

                auto& p = target_base.pose.position;
                auto& o = target_base.pose.orientation;
                p.z = Z_HOVER;  // override z to hover height for safety — only XY should affect reachability
                o.x = 1.0; o.y = 0.0; o.z = 0.0; o.w = 0.0;  // override orientation to fixed downwards-facing for safety

                // ── SAFETY CHECKS ─────────────────────────────────────────────
                double xy_radius = std::sqrt(p.x * p.x + p.y * p.y);

                if (p.z < Z_MIN) {
                    RCLCPP_ERROR(node->get_logger(),
                        "\n"
                        "╔══════════════════════════════════════╗\n"
                        "║  SAFETY ABORT — Z TOO LOW            ║\n"
                        "║  z=%.4f is below floor z=%.4f      ║\n"
                        "║  Check static TF / paper_origin z    ║\n"
                        "╚══════════════════════════════════════╝",
                        p.z, Z_MIN);
                    executing = false;
                    return;
                }

                if (p.z > Z_MAX) {
                    RCLCPP_ERROR(node->get_logger(),
                        "\n"
                        "╔══════════════════════════════════════╗\n"
                        "║  SAFETY ABORT — Z TOO HIGH           ║\n"
                        "║  z=%.4f is above ceiling z=%.4f    ║\n"
                        "╚══════════════════════════════════════╝",
                        p.z, Z_MAX);
                    executing = false;
                    return;
                }

                if (xy_radius > XY_RADIUS_MAX) {
                    RCLCPP_ERROR(node->get_logger(),
                        "\n"
                        "╔══════════════════════════════════════╗\n"
                        "║  SAFETY ABORT — XY OUT OF RANGE      ║\n"
                        "║  radius=%.4f > max=%.4f            ║\n"
                        "╚══════════════════════════════════════╝",
                        xy_radius, XY_RADIUS_MAX);
                    executing = false;
                    return;
                }
                // ── END SAFETY CHECKS ─────────────────────────────────────────

                // Step 2 — print and wait for confirmation
                RCLCPP_INFO(node->get_logger(),
                    "\n"
                    "══════════════════════════════════════\n"
                    "  Target in %s:\n"
                    "    position  x=%.4f  y=%.4f  z=%.4f\n"
                    "    orient    x=%.4f  y=%.4f  z=%.4f  w=%.4f\n"
                    "  Safety OK | xy_radius=%.4f\n"
                    "══════════════════════════════════════\n"
                    "  Press ENTER to execute, Ctrl+C to abort.",
                    TARGET_FRAME.c_str(),
                    p.x, p.y, p.z,
                    o.x, o.y, o.z, o.w,
                    xy_radius
                );

                std::cin.ignore();

                // Step 3 — plan
                RCLCPP_INFO(node->get_logger(), "Planning...");
                arm->setStartStateToCurrentState();
                arm->setPoseTarget(target_base);

                moveit::planning_interface::MoveGroupInterface::Plan plan;
                bool ok = (arm->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

                if (!ok) {
                    RCLCPP_ERROR(node->get_logger(),
                        "Planning FAILED — check pose reachability and TF.");
                    executing = false;
                    return;
                }

                // Step 4 — execute
                RCLCPP_INFO(node->get_logger(), "Executing on hardware...");
                auto result = arm->execute(plan);

                if (result == moveit::core::MoveItErrorCode::SUCCESS) {
                    RCLCPP_INFO(node->get_logger(),
                        "Reached target: x=%.4f y=%.4f z=%.4f",
                        p.x, p.y, p.z);
                } else {
                    RCLCPP_ERROR(node->get_logger(),
                        "Execution FAILED. Error code: %d", result.val);
                }

                executing = false;

            }).detach();
        }
    );

    RCLCPP_INFO(node->get_logger(),
        "\n"
        "─────────────────────────────────────────\n"
        "  arm_target_tester ready\n"
        "  Safety limits active:\n"
        "    Z floor  = %.3f m\n"
        "    Z ceiling= %.3f m\n"
        "    XY radius= %.3f m\n"
        "  Run target_marker_server.py in parallel\n"
        "  Drag marker in RViz, then:\n"
        "  ros2 topic pub /run_to_target std_msgs/msg/Bool \"data: true\" --once\n"
        "─────────────────────────────────────────",
        Z_MIN, Z_MAX, XY_RADIUS_MAX
    );

    spinner.join();
    rclcpp::shutdown();
    return 0;
}