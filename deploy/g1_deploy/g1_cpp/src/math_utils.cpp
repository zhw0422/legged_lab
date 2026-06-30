#include "g1_cpp/math_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace g1 {

namespace {

Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

float dot(const Vec3& a, const Vec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

}  // namespace

Vec3 quat_rotate_inverse_xyzw(const QuatXyzw& q, const Vec3& v) {
  float qw = q[3];
  Vec3 qv{q[0], q[1], q[2]};
  Vec3 b = cross(qv, v);
  float c = dot(qv, v);
  return {
      v[0] * (2.0f * qw * qw - 1.0f) - b[0] * qw * 2.0f + qv[0] * c * 2.0f,
      v[1] * (2.0f * qw * qw - 1.0f) - b[1] * qw * 2.0f + qv[1] * c * 2.0f,
      v[2] * (2.0f * qw * qw - 1.0f) - b[2] * qw * 2.0f + qv[2] * c * 2.0f,
  };
}

Vec3 quat_apply_xyzw(const QuatXyzw& q, const Vec3& v) {
  float qw = q[3];
  Vec3 qv{q[0], q[1], q[2]};
  Vec3 b = cross(qv, v);
  float c = dot(qv, v);
  return {
      v[0] * (2.0f * qw * qw - 1.0f) + b[0] * qw * 2.0f + qv[0] * c * 2.0f,
      v[1] * (2.0f * qw * qw - 1.0f) + b[1] * qw * 2.0f + qv[1] * c * 2.0f,
      v[2] * (2.0f * qw * qw - 1.0f) + b[2] * qw * 2.0f + qv[2] * c * 2.0f,
  };
}

Vec3 compute_projected_gravity(const QuatXyzw& q_xyzw) {
  return quat_rotate_inverse_xyzw(q_xyzw, {0.0f, 0.0f, -1.0f});
}

Vec3 quat_wxyz_to_euler_xyz(const QuatWxyz& q) {
  float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  float w = q[0] / norm, x = q[1] / norm, y = q[2] / norm, z = q[3] / norm;
  float roll = std::atan2(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y));
  float sinp = 2.0f * (w * y - z * x);
  float pitch = std::asin(std::clamp(sinp, -1.0f, 1.0f));
  float yaw = std::atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
  return {roll, pitch, yaw};
}

float yaw_from_xmat(const double* xmat) {
  return static_cast<float>(std::atan2(xmat[3], xmat[0]));
}

std::array<double, 9> yaw_rotation_matrix(double yaw) {
  double c = std::cos(yaw), s = std::sin(yaw);
  return {c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0};
}

std::vector<float> build_proprio_obs(const Vec3& base_ang_vel,
                                     const Vec3& projected_gravity,
                                     const VelocityCommand& commands,
                                     const std::vector<float>& dof_pos_rel,
                                     const std::vector<float>& dof_vel,
                                     const std::vector<float>& last_action,
                                     float ang_vel_scale,
                                     float dof_pos_scale,
                                     float dof_vel_scale,
                                     float last_action_scale,
                                     const std::vector<float>& command_scale) {
  std::vector<float> obs;
  obs.reserve(9 + dof_pos_rel.size() + dof_vel.size() + last_action.size());
  for (float v : base_ang_vel) obs.push_back(v * ang_vel_scale);
  for (float v : projected_gravity) obs.push_back(v);
  obs.push_back(commands.vx * command_scale.at(0));
  obs.push_back(commands.vy * command_scale.at(1));
  obs.push_back(commands.vyaw * command_scale.at(2));
  for (float v : dof_pos_rel) obs.push_back(v * dof_pos_scale);
  for (float v : dof_vel) obs.push_back(v * dof_vel_scale);
  for (float v : last_action) obs.push_back(v * last_action_scale);
  return obs;
}

std::vector<float> stack_history_group_major(const std::vector<std::vector<float>>& history,
                                             int num_joints) {
  if (history.empty()) return {};
  std::vector<std::pair<int, int>> slices = {
      {0, 3},
      {3, 6},
      {6, 9},
      {9, 9 + num_joints},
      {9 + num_joints, 9 + 2 * num_joints},
      {9 + 2 * num_joints, 9 + 3 * num_joints},
  };
  std::vector<float> out;
  out.reserve(history.size() * history.front().size());
  for (auto [start, end] : slices) {
    for (const auto& frame : history) {
      if (end > static_cast<int>(frame.size())) throw std::runtime_error("Observation history frame is too small.");
      out.insert(out.end(), frame.begin() + start, frame.begin() + end);
    }
  }
  return out;
}

}  // namespace g1
