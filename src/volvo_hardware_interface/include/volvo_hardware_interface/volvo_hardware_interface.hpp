#pragma once

#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include <string>
#include <vector>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>

namespace volvo_hardware_interface
{

class VolvoHardwareInterface : public hardware_interface::SystemInterface
{
public:
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  std::vector<hardware_interface::StateInterface>
  export_state_interfaces() override;

  std::vector<hardware_interface::CommandInterface>
  export_command_interfaces() override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  int serial_fd_;
  std::string serial_port_;

  std::vector<double> hw_commands_;
  std::vector<double> hw_states_;
  std::vector<double> hw_velocities_;
  std::vector<double> hw_efforts_;

  // Per-joint PWM limits (µs)
  // Joints 0-2: SG996R → max 2500
  // Joints 3-5: MG90S  → max 2400
  const int PWM_MIN[6] = {500, 500, 500, 500, 500, 500};
  const int PWM_MAX[6] = {2500, 2500, 2500, 2400, 2400, 2400};

  int radiansToDegrees(double radians, int joint_idx);
  bool sendSerial(const std::string & msg);
};

}  // namespace volvo_hardware_interface