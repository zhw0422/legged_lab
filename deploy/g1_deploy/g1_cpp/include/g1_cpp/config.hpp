#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace g1 {

struct Range {
  float lo = 0.0f;
  float hi = 0.0f;
};

struct Config {
  std::filesystem::path source_path;
  std::filesystem::path g1_deploy_dir;

  std::string policy_type = "flat";
  std::string xml_path;
  std::string policy_path;
  float control_dt = 0.02f;
  float sim_dt = 0.005f;
  int history_length = 1;
  int num_actions = 0;
  int num_obs = 0;
  float init_height = 0.90f;

  std::vector<float> kps;
  std::vector<float> kds;
  std::vector<float> default_joint_pos;
  std::vector<float> action_scale;
  std::vector<float> command_scale{1.0f, 1.0f, 1.0f};
  float action_clip = 0.0f;
  bool has_action_clip = false;
  float ang_vel_scale = 1.0f;
  float dof_pos_scale = 1.0f;
  float dof_vel_scale = 1.0f;
  float last_action_scale = 1.0f;

  std::vector<std::string> joint_names_mujoco;
  std::vector<std::string> actuator_names_mujoco;
  std::vector<std::string> sdk_joint_order;
  std::vector<std::string> body_names;
  std::vector<int> mujoco_to_isaac_map;
  std::vector<int> isaac_to_mujoco_map;
  std::vector<int> sdk2isaac_idx;

  std::unordered_map<std::string, Range> command_range;
  std::string gamepad_type_sim2sim = "gamesir";

  std::string terrain_sensor_body = "torso_link";
  std::vector<float> terrain_sensor_offset{0.0f, 0.0f, 20.0f};
  int terrain_map_length = 0;
  int terrain_map_width = 0;
  int terrain_map_coord_dim = 3;
  std::vector<float> terrain_map_size{1.6f, 1.0f};
  std::vector<float> terrain_map_z_clip{-1.2f, 0.0f};
  std::string terrain_ordering = "xy";
  int terrain_flg_static = 1;
  int terrain_bodyexclude = -1;

  bool include_state_estimation = true;
  std::string anchor_body_name;
  int motion_total_steps = 0;

  std::string msg_type = "hg";
  std::string lowcmd_topic = "rt/lowcmd";
  std::string lowstate_topic = "rt/lowstate";
};

std::filesystem::path resolve_config_path(const std::filesystem::path& g1_deploy_dir,
                                          const std::string& name);
std::filesystem::path resolve_asset_path(const std::filesystem::path& g1_deploy_dir,
                                         const std::string& value);
std::filesystem::path resolve_policy_path(const std::filesystem::path& g1_deploy_dir,
                                          const std::string& value,
                                          const std::string& override_model);
Config load_config(const std::filesystem::path& path,
                   const std::filesystem::path& g1_deploy_dir);

}  // namespace g1
