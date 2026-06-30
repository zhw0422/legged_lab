#include "g1_cpp/config.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace fs = std::filesystem;

namespace g1 {

namespace {

template <typename T>
std::vector<T> as_vec(const YAML::Node& node) {
  std::vector<T> out;
  if (!node) return out;
  if (!node.IsSequence()) {
    out.push_back(node.as<T>());
    return out;
  }
  out.reserve(node.size());
  for (const auto& v : node) out.push_back(v.as<T>());
  return out;
}

std::string as_string(const YAML::Node& node, const std::string& fallback = "") {
  return node ? node.as<std::string>() : fallback;
}

float as_float(const YAML::Node& node, float fallback = 0.0f) {
  return node ? node.as<float>() : fallback;
}

int as_int(const YAML::Node& node, int fallback = 0) {
  return node ? node.as<int>() : fallback;
}

bool as_bool(const YAML::Node& node, bool fallback = false) {
  return node ? node.as<bool>() : fallback;
}

Range parse_range(const YAML::Node& node, Range fallback) {
  if (!node || !node.IsSequence() || node.size() != 2) return fallback;
  return {node[0].as<float>(), node[1].as<float>()};
}

}  // namespace

fs::path resolve_config_path(const fs::path& g1_deploy_dir, const std::string& name) {
  fs::path p(name);
  if (fs::exists(p)) return fs::absolute(p);
  if (!p.empty() && p.begin()->string() == "config") return g1_deploy_dir / p;
  return g1_deploy_dir / "config" / p;
}

fs::path resolve_asset_path(const fs::path& g1_deploy_dir, const std::string& value) {
  fs::path p(value);
  if (p.is_absolute() && fs::exists(p)) return p;
  fs::path direct = g1_deploy_dir / p;
  if (fs::exists(direct)) return direct;
  fs::path asset = g1_deploy_dir / "assets" / p.filename();
  if (fs::exists(asset)) return asset;
  return direct;
}

fs::path resolve_policy_path(const fs::path& g1_deploy_dir,
                             const std::string& value,
                             const std::string& override_model) {
  fs::path selected = override_model.empty() ? fs::path(value) : fs::path(override_model);
  if (selected.is_absolute() && fs::exists(selected)) return selected;
  fs::path direct = g1_deploy_dir / selected;
  if (fs::exists(direct)) return direct;
  return g1_deploy_dir / "exported_policy" / selected.filename();
}

Config load_config(const fs::path& path, const fs::path& g1_deploy_dir) {
  YAML::Node y = YAML::LoadFile(path.string());
  Config c;
  c.source_path = path;
  c.g1_deploy_dir = g1_deploy_dir;

  c.policy_type = as_string(y["policy_type"], c.policy_type);
  c.xml_path = as_string(y["xml_path"]);
  c.policy_path = as_string(y["policy_path"]);
  c.control_dt = as_float(y["control_dt"], c.control_dt);
  c.sim_dt = as_float(y["sim_dt"], c.sim_dt);
  c.history_length = as_int(y["history_length"], c.history_length);
  c.num_actions = as_int(y["num_actions"], c.num_actions);
  c.num_obs = as_int(y["num_obs"], c.num_obs);
  c.init_height = as_float(y["init_height"], c.init_height);

  c.kps = as_vec<float>(y["kps"]);
  c.kds = as_vec<float>(y["kds"]);
  c.default_joint_pos = as_vec<float>(y["default_joint_pos"]);
  c.action_scale = as_vec<float>(y["action_scale"]);
  if (c.action_scale.size() == 1 && c.num_actions > 1) {
    c.action_scale.assign(static_cast<size_t>(c.num_actions), c.action_scale.front());
  }
  if (c.action_scale.empty() && c.num_actions > 0) {
    c.action_scale.assign(static_cast<size_t>(c.num_actions), 1.0f);
  }
  c.command_scale = as_vec<float>(y["command_scale"]);
  if (c.command_scale.empty()) c.command_scale = {1.0f, 1.0f, 1.0f};
  c.has_action_clip = static_cast<bool>(y["action_clip"]);
  c.action_clip = as_float(y["action_clip"], c.action_clip);
  c.ang_vel_scale = as_float(y["ang_vel_scale"], c.ang_vel_scale);
  c.dof_pos_scale = as_float(y["dof_pos_scale"], c.dof_pos_scale);
  c.dof_vel_scale = as_float(y["dof_vel_scale"], c.dof_vel_scale);
  c.last_action_scale = as_float(y["last_action_scale"], c.last_action_scale);

  c.joint_names_mujoco = as_vec<std::string>(y["joint_names_mujoco"]);
  c.actuator_names_mujoco = as_vec<std::string>(y["actuator_names_mujoco"]);
  c.sdk_joint_order = as_vec<std::string>(y["sdk_joint_order"]);
  c.body_names = as_vec<std::string>(y["body_names"]);
  c.mujoco_to_isaac_map = as_vec<int>(y["mujoco_to_isaac_map"]);
  c.isaac_to_mujoco_map = as_vec<int>(y["isaac_to_mujoco_map"]);
  c.sdk2isaac_idx = as_vec<int>(y["sdk2isaac_idx"]);

  c.command_range["lin_vel_x"] = parse_range(y["command_range"]["lin_vel_x"], {-1.0f, 1.0f});
  c.command_range["lin_vel_y"] = parse_range(y["command_range"]["lin_vel_y"], {-0.5f, 0.5f});
  c.command_range["ang_vel_z"] = parse_range(y["command_range"]["ang_vel_z"], {-1.0f, 1.0f});
  c.gamepad_type_sim2sim = as_string(y["gamepad_type_sim2sim"], c.gamepad_type_sim2sim);

  c.terrain_sensor_body = as_string(y["terrain_sensor_body"], c.terrain_sensor_body);
  c.terrain_sensor_offset = as_vec<float>(y["terrain_sensor_offset"]);
  if (c.terrain_sensor_offset.size() != 3) c.terrain_sensor_offset = {0.0f, 0.0f, 20.0f};
  c.terrain_map_length = as_int(y["terrain_map_length"], c.terrain_map_length);
  c.terrain_map_width = as_int(y["terrain_map_width"], c.terrain_map_width);
  c.terrain_map_coord_dim = as_int(y["terrain_map_coord_dim"], c.terrain_map_coord_dim);
  c.terrain_map_size = as_vec<float>(y["terrain_map_size"]);
  if (c.terrain_map_size.size() != 2) c.terrain_map_size = {1.6f, 1.0f};
  c.terrain_map_z_clip = as_vec<float>(y["terrain_map_z_clip"]);
  if (c.terrain_map_z_clip.size() != 2) c.terrain_map_z_clip = {-1.2f, 0.0f};
  c.terrain_ordering = as_string(y["terrain_ordering"], c.terrain_ordering);
  c.terrain_flg_static = as_int(y["terrain_flg_static"], c.terrain_flg_static);
  c.terrain_bodyexclude = as_int(y["terrain_bodyexclude"], c.terrain_bodyexclude);

  c.include_state_estimation = as_bool(y["include_state_estimation"], c.include_state_estimation);
  c.anchor_body_name = as_string(y["anchor_body_name"], c.anchor_body_name);
  c.motion_total_steps = as_int(y["motion_total_steps"], c.motion_total_steps);
  c.msg_type = as_string(y["msg_type"], c.msg_type);
  c.lowcmd_topic = as_string(y["lowcmd_topic"], c.lowcmd_topic);
  c.lowstate_topic = as_string(y["lowstate_topic"], c.lowstate_topic);

  if (c.xml_path.empty()) throw std::runtime_error("Config is missing xml_path: " + path.string());
  if (c.num_actions <= 0) c.num_actions = static_cast<int>(c.default_joint_pos.size());
  return c;
}

}  // namespace g1
