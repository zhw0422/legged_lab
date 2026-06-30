#pragma once

#include "g1_cpp/controllers.hpp"

#include <array>
#include <vector>

namespace g1 {

using Vec3 = std::array<float, 3>;
using QuatWxyz = std::array<float, 4>;
using QuatXyzw = std::array<float, 4>;

Vec3 quat_rotate_inverse_xyzw(const QuatXyzw& q, const Vec3& v);
Vec3 quat_apply_xyzw(const QuatXyzw& q, const Vec3& v);
Vec3 compute_projected_gravity(const QuatXyzw& q_xyzw);
Vec3 quat_wxyz_to_euler_xyz(const QuatWxyz& q);
float yaw_from_xmat(const double* xmat);
std::array<double, 9> yaw_rotation_matrix(double yaw);
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
                                     const std::vector<float>& command_scale);
std::vector<float> stack_history_group_major(const std::vector<std::vector<float>>& history,
                                             int num_joints);

}  // namespace g1
