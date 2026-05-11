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

    // ── THIS was the broken line — angle bracket missing in previous send ────
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

                // Step 2 — print and wait for confirmation
                RCLCPP_INFO(node->get_logger(),
                    "\n"
                    "══════════════════════════════════════\n"
                    "  Target in %s:\n"
                    "    position  x=%.4f  y=%.4f  z=%.4f\n"
                    "    orient    x=%.4f  y=%.4f  z=%.4f  w=%.4f\n"
                    "══════════════════════════════════════\n"
                    "  Press ENTER to execute, Ctrl+C to abort.",
                    TARGET_FRAME.c_str(),
                    p.x, p.y, p.z,
                    o.x, o.y, o.z, o.w
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
                        "Planning FAILED.\n"
                        "  Check: pose reachable? TF broadcasting? Orientation valid?");
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
        "  Run target_marker_server.py in parallel\n"
        "  Drag marker in RViz, then:\n"
        "  ros2 topic pub /run_to_target std_msgs/msg/Bool \"data: true\" --once\n"
        "─────────────────────────────────────────"
    );

    spinner.join();
    rclcpp::shutdown();
    return 0;
}