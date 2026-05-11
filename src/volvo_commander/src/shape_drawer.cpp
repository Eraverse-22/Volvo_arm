/**
 * shape_drawer.cpp  —  FIXED v3
 * ================================
 * ROS2 / MoveIt2 node that executes a drawing motion from /draw_poses.
 *
 * FIXES IN THIS VERSION (vs original broken code)
 * ─────────────────────────────────────────────────
 *
 * FIX-1  Per-stroke Cartesian planning.
 *        ORIGINAL: computeCartesianPath called on ALL poses in one shot,
 *        including hover poses interleaved between strokes → pen dragged
 *        through hover positions at paper height → chaotic scratches.
 *        FIXED: iterate over strokes using stroke_ranges from /draw_strokes.
 *        For each stroke, extract ONLY draw poses (isHover() == false),
 *        run computeCartesianPath on that subset independently.
 *
 * FIX-2  Subscribe to /draw_strokes.
 *        ORIGINAL: /draw_strokes topic was published by shape_planner but
 *        shape_drawer had no subscription → stroke ranges silently discarded.
 *        FIXED: strokesCallback stores (start,end) index pairs in stroke_ranges_.
 *
 * FIX-3  4-phase motion sequence per stroke.
 *        ORIGINAL: moved to approach_pose (z+3cm) then jumped directly to
 *        paper-level Cartesian path — no descent, causing impact or MoveIt
 *        timeout on approach planning.
 *        FIXED: per stroke:
 *          Phase 1 — Joint-space move: hover above stroke[0]     (approach)
 *          Phase 2 — Joint-space move: stroke[0] at paper_z      (descend)
 *          Phase 3 — Cartesian path:   draw_poses only           (draw)
 *          Phase 4 — Joint-space move: hover above stroke[-1]    (lift)
 *        Phase 4 ALWAYS executes, even if Phase 3 failed — pen never
 *        left pressing against the paper.
 *
 * FIX-4  ROS2 tunable parameters.
 *        ORIGINAL: vel_scale, acc_scale hardcoded at 0.1.
 *        FIXED: all motion parameters exposed as ROS2 parameters:
 *          vel_scale               (default 0.10)
 *          acc_scale               (default 0.10)
 *          cartesian_eef_step      (default 0.003 m)
 *          min_cartesian_fraction  (default 0.90)
 *          hover_z_offset          (default 0.035 m)
 *          planning_timeout        (default 5.0 s)
 *
 * FIX-5  Publish /arm_done and /draw_complete.
 *        ORIGINAL: /arm_done never published → shape_detection_node
 *        stayed locked after first draw, never re-arming.
 *        FIXED: /arm_done (Bool=true) published after all strokes complete.
 *        /draw_complete (Bool=true/false) indicates overall success/failure.
 *
 * SUBSCRIBE TO /draw_metadata for paper_z and hover_z values so this node
 * does not need to duplicate those parameters from shape_planner.
 *
 * TOPICS
 *   Sub:  /draw_poses      (geometry_msgs/PoseArray)
 *         /draw_strokes    (std_msgs/Float32MultiArray)  [start,end pairs]
 *         /draw_metadata   (std_msgs/Float32MultiArray)  [paper_z,hover_z,n]
 *         /execute_draw    (std_msgs/Bool)
 *   Pub:  /arm_done        (std_msgs/Bool)
 *         /draw_complete   (std_msgs/Bool)
 *
 * HOVER POSE DETECTION
 *   shape_planner marks hover poses with orientation.w = -0.001.
 *   isHover() checks: pose.orientation.w < -0.0005
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <thread>
#include <mutex>
#include <vector>
#include <memory>
#include <cmath>
#include <string>

using Pose = geometry_msgs::msg::Pose;

// ── Hover pose detection ───────────────────────────────────────────────────────
/**
 * shape_planner marks hover poses with orientation.w = -0.001.
 * Any negative w is our sentinel — safe because unit quaternions for
 * downward orientation have w=0, so w < -0.0005 is unambiguous.
 */
static bool isHover(const Pose& p)
{
    return p.orientation.w < -0.0005;
}

/**
 * Rebuild a valid downward orientation from a hover sentinel pose.
 * The position is kept; orientation is replaced with the true draw quaternion.
 */
static Pose makeDrawPose(const Pose& p)
{
    Pose out  = p;
    out.orientation.x = 1.0;
    out.orientation.y = 0.0;
    out.orientation.z = 0.0;
    out.orientation.w = 0.0;
    return out;
}

// ══════════════════════════════════════════════════════════════════════════════
class ShapeDrawer : public rclcpp::Node
{
public:
    ShapeDrawer() : Node("shape_drawer_node")
    {
        // ── Declare parameters (tunable at runtime) ────────────────────────
        this->declare_parameter("planning_group",       std::string("volvo_arm"));
        this->declare_parameter("end_effector_link",    std::string("EE"));
        this->declare_parameter("vel_scale",            0.10);
        this->declare_parameter("acc_scale",            0.10);
        this->declare_parameter("cartesian_eef_step",   0.003);
        this->declare_parameter("min_cartesian_fraction", 0.90);
        this->declare_parameter("hover_z_offset",       0.035);
        this->declare_parameter("planning_timeout",     5.0);

        // ── Subscriptions ──────────────────────────────────────────────────
        sub_poses_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/draw_poses", 10,
            std::bind(&ShapeDrawer::posesCallback, this, std::placeholders::_1));

        // FIX-2: subscribe to /draw_strokes
        sub_strokes_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/draw_strokes", 10,
            std::bind(&ShapeDrawer::strokesCallback, this, std::placeholders::_1));

        // Subscribe to /draw_metadata to get paper_z and hover_z
        sub_metadata_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/draw_metadata", 10,
            std::bind(&ShapeDrawer::metadataCallback, this, std::placeholders::_1));

        sub_execute_ = this->create_subscription<std_msgs::msg::Bool>(
            "/execute_draw", 10,
            std::bind(&ShapeDrawer::executeCallback, this, std::placeholders::_1));

        // ── Publishers ─────────────────────────────────────────────────────
        // FIX-5: publish /arm_done and /draw_complete
        pub_arm_done_     = this->create_publisher<std_msgs::msg::Bool>("/arm_done",      10);
        pub_draw_complete_ = this->create_publisher<std_msgs::msg::Bool>("/draw_complete", 10);

        RCLCPP_INFO(this->get_logger(),
            "ShapeDrawer created. Call init_move_group() after shared_ptr exists.");
    }

    /**
     * MUST be called after the node's shared_ptr is created (i.e. after
     * std::make_shared<ShapeDrawer>() returns), because MoveGroupInterface
     * calls shared_from_this() internally.
     */
    void init_move_group()
    {
        auto group = this->get_parameter("planning_group").as_string();
        auto eef   = this->get_parameter("end_effector_link").as_string();

        move_group_ = std::make_shared<
            moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), group);

        move_group_->setEndEffectorLink(eef);
        applyMotionParameters();

        RCLCPP_INFO(this->get_logger(),
            "MoveGroupInterface ready | group=%s | eef=%s | "
            "vel=%.2f | acc=%.2f | eef_step=%.4f | min_frac=%.2f | timeout=%.1f s",
            group.c_str(), eef.c_str(),
            this->get_parameter("vel_scale").as_double(),
            this->get_parameter("acc_scale").as_double(),
            this->get_parameter("cartesian_eef_step").as_double(),
            this->get_parameter("min_cartesian_fraction").as_double(),
            this->get_parameter("planning_timeout").as_double());
    }

private:
    // ── Callbacks ──────────────────────────────────────────────────────────────

    void posesCallback(
        const geometry_msgs::msg::PoseArray::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        all_poses_ = msg->poses;
        RCLCPP_INFO(this->get_logger(),
            "Received %zu poses (draw + hover combined).", all_poses_.size());
    }

    // FIX-2: store stroke ranges
    void strokesCallback(
        const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        stroke_ranges_.clear();
        for (auto v : msg->data)
            stroke_ranges_.push_back(static_cast<int>(v));
        RCLCPP_INFO(this->get_logger(),
            "Received %zu stroke ranges (%zu strokes).",
            stroke_ranges_.size(), stroke_ranges_.size() / 2);
    }

    void metadataCallback(
        const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        if (msg->data.size() >= 3) {
            std::lock_guard<std::mutex> lock(data_mutex_);
            meta_paper_z_  = static_cast<double>(msg->data[0]);
            meta_hover_z_  = static_cast<double>(msg->data[1]);
            meta_n_strokes_ = static_cast<int>(msg->data[2]);
            RCLCPP_INFO(this->get_logger(),
                "Metadata received: paper_z=%.4f m | hover_z=%.4f m | n_strokes=%d",
                meta_paper_z_, meta_hover_z_, meta_n_strokes_);
        }
    }

    void executeCallback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (!msg->data) return;

        std::lock_guard<std::mutex> lock(data_mutex_);

        if (all_poses_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No poses to execute — ignoring.");
            return;
        }
        if (stroke_ranges_.empty()) {
            RCLCPP_WARN(this->get_logger(),
                "No stroke ranges — /draw_strokes not yet received. "
                "Wait for shape_planner to publish before executing.");
            return;
        }
        if (executing_) {
            RCLCPP_WARN(this->get_logger(), "Already executing — ignoring.");
            return;
        }

        executing_ = true;
        // Launch in a detached thread (MoveIt calls cannot block the ROS spin thread)
        std::thread(
            &ShapeDrawer::executeAllStrokes,
            this,
            all_poses_,
            stroke_ranges_
        ).detach();
    }

    // ── Core execution loop ────────────────────────────────────────────────────

    /**
     * FIX-1 + FIX-3: Per-stroke 4-phase execution.
     *
     * For each stroke (defined by [start_idx, end_idx] from stroke_ranges):
     *   1. Extract only DRAW poses (skip hover sentinels)
     *   2. Phase 1: joint-space approach → hover above stroke[0]
     *   3. Phase 2: joint-space descend  → stroke[0] at paper_z
     *   4. Phase 3: Cartesian draw       → through all draw poses
     *   5. Phase 4: joint-space lift     → hover above stroke[-1]  (ALWAYS)
     */
    void executeAllStrokes(
        const std::vector<Pose> poses,
        const std::vector<int>  stroke_ranges)
    {
        applyMotionParameters();  // re-apply in case params changed at runtime

        int n_strokes = static_cast<int>(stroke_ranges.size()) / 2;
        RCLCPP_INFO(this->get_logger(),
            "Starting execution: %d strokes.", n_strokes);

        double hover_z_offset = this->get_parameter("hover_z_offset").as_double();
        // Prefer hover_z from metadata if available, else compute from paper_z
        double hover_z = (meta_hover_z_ != 0.0)
                         ? meta_hover_z_
                         : meta_paper_z_ + hover_z_offset;

        int success_count = 0;

        for (int si = 0; si < n_strokes; ++si) {
            int start_idx = stroke_ranges[si * 2];
            int end_idx   = stroke_ranges[si * 2 + 1];

            // Validate indices
            if (start_idx < 0 || end_idx >= static_cast<int>(poses.size())
                || start_idx > end_idx)
            {
                RCLCPP_ERROR(this->get_logger(),
                    "Stroke %d: invalid range [%d, %d] (total poses=%zu). Skipping.",
                    si, start_idx, end_idx, poses.size());
                continue;
            }

            // Extract DRAW-only poses for this stroke (skip hover sentinels)
            std::vector<Pose> draw_poses;
            for (int pi = start_idx; pi <= end_idx; ++pi) {
                if (!isHover(poses[pi]))
                    draw_poses.push_back(poses[pi]);
            }

            if (draw_poses.empty()) {
                RCLCPP_WARN(this->get_logger(),
                    "Stroke %d: no draw poses after filtering. Skipping.", si);
                continue;
            }

            RCLCPP_INFO(this->get_logger(),
                "--- Stroke %d/%d: %zu draw poses ---", si+1, n_strokes, draw_poses.size());

            bool stroke_ok = executeOneStroke(draw_poses, hover_z, si);
            if (stroke_ok) ++success_count;
        }

        // FIX-5: publish completion feedback
        bool all_ok = (success_count == n_strokes);

        auto arm_done_msg = std_msgs::msg::Bool();
        arm_done_msg.data = true;
        pub_arm_done_->publish(arm_done_msg);

        auto complete_msg = std_msgs::msg::Bool();
        complete_msg.data = all_ok;
        pub_draw_complete_->publish(complete_msg);

        RCLCPP_INFO(this->get_logger(),
            "Execution complete: %d/%d strokes succeeded. "
            "/arm_done published.", success_count, n_strokes);

        executing_ = false;
    }

    /**
     * FIX-3: 4-phase motion for a single stroke.
     *
     * Phase 1 — Approach:  joint-space to hover position above draw_poses[0]
     * Phase 2 — Descend:   joint-space to draw_poses[0] (first contact point)
     * Phase 3 — Draw:      computeCartesianPath through all draw_poses
     * Phase 4 — Lift:      joint-space to hover position above draw_poses.back()
     *
     * Phase 4 ALWAYS executes — pen is always lifted even on failure.
     * Returns true if all 4 phases succeeded.
     */
    bool executeOneStroke(
        const std::vector<Pose>& draw_poses,
        double hover_z,
        int stroke_index)
    {
        bool success = true;
        double timeout = this->get_parameter("planning_timeout").as_double();

        // ── Phase 1: Approach (hover above stroke start) ───────────────────
        Pose approach_pose        = draw_poses.front();
        approach_pose.position.z  = hover_z;
        // Restore valid downward quaternion (in case it was a hover sentinel)
        approach_pose = makeDrawPose(approach_pose);

        RCLCPP_INFO(this->get_logger(),
            "[Stroke %d] Phase 1/4: Approach → (%.3f, %.3f, %.3f)",
            stroke_index,
            approach_pose.position.x,
            approach_pose.position.y,
            approach_pose.position.z);

        move_group_->setPlanningTime(timeout);
        move_group_->setPoseTarget(approach_pose);
        auto res = move_group_->move();
        if (res != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(),
                "[Stroke %d] Phase 1 FAILED (approach). Error: %d. Skipping stroke.",
                stroke_index, res.val);
            return false;  // cannot descend safely if approach failed
        }

        // ── Phase 2: Descend (joint-space to first draw point) ────────────
        Pose start_pose = makeDrawPose(draw_poses.front());
        RCLCPP_INFO(this->get_logger(),
            "[Stroke %d] Phase 2/4: Descend → (%.3f, %.3f, %.3f)",
            stroke_index,
            start_pose.position.x,
            start_pose.position.y,
            start_pose.position.z);

        move_group_->setPlanningTime(timeout);
        move_group_->setPoseTarget(start_pose);
        res = move_group_->move();
        if (res != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(),
                "[Stroke %d] Phase 2 FAILED (descend). Error: %d. Lifting pen.",
                stroke_index, res.val);
            success = false;
            // Fall through to Phase 4 to lift the pen
        }

        // ── Phase 3: Draw (Cartesian path through draw points) ────────────
        if (success) {
            RCLCPP_INFO(this->get_logger(),
                "[Stroke %d] Phase 3/4: Cartesian draw (%zu waypoints).",
                stroke_index, draw_poses.size());

            // Build clean waypoints (all with valid draw quaternion)
            std::vector<Pose> waypoints;
            waypoints.reserve(draw_poses.size());
            for (const auto& p : draw_poses)
                waypoints.push_back(makeDrawPose(p));

            moveit_msgs::msg::RobotTrajectory trajectory;
            double eef_step  = this->get_parameter("cartesian_eef_step").as_double();
            double min_frac  = this->get_parameter("min_cartesian_fraction").as_double();

            double fraction = move_group_->computeCartesianPath(
                waypoints,
                eef_step,
                0.0,          // jump_threshold = 0 disables jump detection
                trajectory);

            RCLCPP_INFO(this->get_logger(),
                "[Stroke %d] Cartesian path fraction: %.3f (min=%.2f)",
                stroke_index, fraction, min_frac);

            if (fraction >= min_frac) {
                res = move_group_->execute(trajectory);
                if (res != moveit::core::MoveItErrorCode::SUCCESS) {
                    RCLCPP_ERROR(this->get_logger(),
                        "[Stroke %d] Phase 3 execute FAILED. Error: %d.",
                        stroke_index, res.val);
                    success = false;
                }
            } else {
                RCLCPP_ERROR(this->get_logger(),
                    "[Stroke %d] Phase 3 SKIPPED: fraction %.3f < minimum %.2f. "
                    "Path likely crosses singularity or exceeds joint limits.",
                    stroke_index, fraction, min_frac);
                success = false;
            }
        }

        // ── Phase 4: Lift (ALWAYS executes) ──────────────────────────────
        Pose lift_pose       = makeDrawPose(draw_poses.back());
        lift_pose.position.z = hover_z;

        RCLCPP_INFO(this->get_logger(),
            "[Stroke %d] Phase 4/4: Lift → (%.3f, %.3f, %.3f) [ALWAYS]",
            stroke_index,
            lift_pose.position.x,
            lift_pose.position.y,
            lift_pose.position.z);

        move_group_->setPlanningTime(timeout);
        move_group_->setPoseTarget(lift_pose);
        res = move_group_->move();
        if (res != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(),
                "[Stroke %d] Phase 4 FAILED (lift). Error: %d. "
                "WARNING: pen may still be on paper!",
                stroke_index, res.val);
            success = false;
        }

        if (success) {
            RCLCPP_INFO(this->get_logger(),
                "[Stroke %d] Complete — all 4 phases succeeded.", stroke_index);
        } else {
            RCLCPP_WARN(this->get_logger(),
                "[Stroke %d] Completed with failures — check logs above.", stroke_index);
        }
        return success;
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    void applyMotionParameters()
    {
        if (!move_group_) return;
        double vel = this->get_parameter("vel_scale").as_double();
        double acc = this->get_parameter("acc_scale").as_double();
        move_group_->setMaxVelocityScalingFactor(vel);
        move_group_->setMaxAccelerationScalingFactor(acc);
    }

    // ── Members ────────────────────────────────────────────────────────────────
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr       sub_poses_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr     sub_strokes_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr     sub_metadata_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr                  sub_execute_;

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr  pub_arm_done_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr  pub_draw_complete_;

    std::vector<Pose>  all_poses_;
    std::vector<int>   stroke_ranges_;
    std::mutex         data_mutex_;
    bool               executing_ = false;

    // Metadata from /draw_metadata
    double meta_paper_z_   = 0.0;
    double meta_hover_z_   = 0.0;
    int    meta_n_strokes_ = 0;
};

// ══════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ShapeDrawer>();

    // init_move_group() MUST be called AFTER make_shared returns
    // because MoveGroupInterface calls shared_from_this() internally.
    node->init_move_group();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}