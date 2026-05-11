#include "volvo_hardware_interface/volvo_hardware_interface.hpp"
#include <pluginlib/class_list_macros.hpp>

#include <cmath>
#include <sstream>
#include <chrono>
#include <thread>

namespace volvo_hardware_interface
{

int VolvoHardwareInterface::radiansToDegrees(double radians, int joint_idx)
{
  const double offset_deg[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // tune per joint
  const int direction[6] = {1, 1, -1, 1, 1, 1};

  double nominal = radians * (180.0 / M_PI);

  // Direction flips the motion, offset shifts the center — both inside direction
  double corrected = direction[joint_idx] * (nominal + offset_deg[joint_idx]);

  // Re-center: inverted joints output negative, map back to 0-180
  if (direction[joint_idx] == -1) {
    corrected = 180.0 + corrected; // flip around 90°
  }

  return std::max(0, std::min(180, static_cast<int>(corrected)));
}
bool VolvoHardwareInterface::sendSerial(const std::string & msg)
{
  if (serial_fd_ < 0) return false;
  ssize_t written = ::write(serial_fd_, msg.c_str(), msg.size());
  return written == static_cast<ssize_t>(msg.size());
}

hardware_interface::CallbackReturn VolvoHardwareInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  serial_port_ = info_.hardware_parameters.count("serial_port") ?
    info_.hardware_parameters.at("serial_port") : "/dev/ttyUSB0";

  hw_commands_.resize(info_.joints.size(), 0.0);
  hw_states_.resize(info_.joints.size(), 0.0);
  hw_velocities_.resize(info_.joints.size(), 0.0);
  hw_efforts_.resize(info_.joints.size(), 0.0);
  serial_fd_ = -1;

  RCLCPP_INFO(rclcpp::get_logger("VolvoHardwareInterface"),
    "Initialized. Serial port: %s", serial_port_.c_str());

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
VolvoHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> interfaces;
  for (size_t i = 0; i < info_.joints.size(); i++) {
    interfaces.emplace_back(
      info_.joints[i].name,
      hardware_interface::HW_IF_POSITION,
      &hw_states_[i]);
    interfaces.emplace_back(
      info_.joints[i].name,
      hardware_interface::HW_IF_VELOCITY,
      &hw_velocities_[i]);
    interfaces.emplace_back(
      info_.joints[i].name,
      hardware_interface::HW_IF_EFFORT,
      &hw_efforts_[i]);
  }
  return interfaces;
}

std::vector<hardware_interface::CommandInterface>
VolvoHardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> interfaces;
  for (size_t i = 0; i < info_.joints.size(); i++) {
    interfaces.emplace_back(
      info_.joints[i].name,
      hardware_interface::HW_IF_POSITION,
      &hw_commands_[i]);
  }
  return interfaces;
}

hardware_interface::CallbackReturn VolvoHardwareInterface::on_activate(
  const rclcpp_lifecycle::State &)
{
  // Open serial port
  serial_fd_ = open(serial_port_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
  if (serial_fd_ < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("VolvoHardwareInterface"),
      "Failed to open serial port: %s", serial_port_.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Configure 115200 8N1
  struct termios tty;
  memset(&tty, 0, sizeof tty);
  tcgetattr(serial_fd_, &tty);
  cfsetospeed(&tty, B115200);
  cfsetispeed(&tty, B115200);
  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~(PARENB | PARODD | CSTOPB | CRTSCTS);
  tty.c_iflag &= ~(IXON | IXOFF | IXANY);
  tty.c_lflag = 0;
  tty.c_oflag = 0;
  tty.c_cc[VMIN]  = 0;
  tty.c_cc[VTIME] = 5;
  tcsetattr(serial_fd_, TCSANOW, &tty);

  // Wait for ESP32 to boot and send READY
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  // Initialize commands to SRDF home position
  const double home_rad[6] = {0.0, 1.57, 1.57, 1.57, 1.57, 1.57};
  for (size_t i = 0; i < hw_commands_.size(); i++) {
    hw_commands_[i] = home_rad[i];
    hw_states_[i]   = home_rad[i];
  }

  // Send home position in degrees to ESP32
  std::ostringstream cmd;
  cmd << "A";
  for (size_t i = 0; i < 6; i++) {
    cmd << radiansToDegrees(home_rad[i], static_cast<int>(i));
    if (i < 5) cmd << ",";
  }
  cmd << "\n";
  sendSerial(cmd.str());

  RCLCPP_INFO(rclcpp::get_logger("VolvoHardwareInterface"),
    "Home command sent (degrees): %s", cmd.str().c_str());

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  RCLCPP_INFO(rclcpp::get_logger("VolvoHardwareInterface"),
    "Serial port opened. Arm at home position.");

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn VolvoHardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  if (serial_fd_ >= 0) {
    sendSerial("HOME\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    close(serial_fd_);
    serial_fd_ = -1;
  }
  RCLCPP_INFO(rclcpp::get_logger("VolvoHardwareInterface"),
    "Serial port closed.");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type VolvoHardwareInterface::read(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  // No feedback from servos — mirror commands back as states
  hw_states_ = hw_commands_;
  for (size_t i = 0; i < hw_velocities_.size(); i++) {
    hw_velocities_[i] = 0.0;
    hw_efforts_[i] = 0.0;
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type VolvoHardwareInterface::write(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  // Build serial command: A<deg1>,<deg2>,...,<deg6>\n
  std::ostringstream cmd;
  cmd << "A";
  for (size_t i = 0; i < hw_commands_.size(); i++) {
    cmd << radiansToDegrees(hw_commands_[i], static_cast<int>(i));
    if (i < hw_commands_.size() - 1) cmd << ",";
  }
  cmd << "\n";

  if (!sendSerial(cmd.str())) {
    RCLCPP_WARN(rclcpp::get_logger("VolvoHardwareInterface"),
      "Serial write failed");
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

}  // namespace volvo_hardware_interface

PLUGINLIB_EXPORT_CLASS(
  volvo_hardware_interface::VolvoHardwareInterface,
  hardware_interface::SystemInterface)