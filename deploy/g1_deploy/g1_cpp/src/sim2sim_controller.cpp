#include "g1_cpp/sim2sim_controller.hpp"

#include "g1_cpp/math_utils.hpp"

#ifdef G1_CPP_HAS_GLFW
#include <GLFW/glfw3.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;

namespace g1 {

namespace {

double wall_time() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

template <typename T>
std::vector<T> reorder(const std::vector<T>& v, const std::vector<int>& map) {
  std::vector<T> out(map.size());
  for (size_t i = 0; i < map.size(); ++i) out[i] = v.at(static_cast<size_t>(map[i]));
  return out;
}

std::vector<float> subtract(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] - b[i];
  return out;
}

QuatXyzw wxyz_to_xyzw(const double* q) {
  return {static_cast<float>(q[1]), static_cast<float>(q[2]), static_cast<float>(q[3]), static_cast<float>(q[0])};
}

QuatWxyz quat_inv_wxyz(const QuatWxyz& q) {
  return {q[0], -q[1], -q[2], -q[3]};
}

QuatWxyz quat_mul_wxyz(const QuatWxyz& a, const QuatWxyz& b) {
  return {
      a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  };
}

Vec3 quat_apply_wxyz(const QuatWxyz& q, const Vec3& v) {
  return quat_apply_xyzw({q[1], q[2], q[3], q[0]}, v);
}

std::array<float, 9> matrix_from_quat_wxyz(const QuatWxyz& q) {
  float w = q[0], x = q[1], y = q[2], z = q[3];
  return {
      1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w), 2.0f * (x * z + y * w),
      2.0f * (x * y + z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - x * w),
      2.0f * (x * z - y * w), 2.0f * (y * z + x * w), 1.0f - 2.0f * (x * x + y * y),
  };
}

Range cfg_range(const Config& cfg, const std::string& key, Range fallback) {
  auto it = cfg.command_range.find(key);
  return it == cfg.command_range.end() ? fallback : it->second;
}

void append(std::vector<float>& dst, const std::vector<float>& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

void append3(std::vector<float>& dst, const Vec3& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

#ifdef G1_CPP_HAS_GLFW
class MujocoWindow {
 public:
  MujocoWindow(mjModel* model, mjData* data, bool disabled) : model_(model), data_(data), disabled_(disabled) {
    if (disabled_) return;
    if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW.");
    window_ = glfwCreateWindow(1280, 900, "G1 C++ MuJoCo Sim2Sim", nullptr, nullptr);
    if (!window_) throw std::runtime_error("Failed to create GLFW window.");
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0);
    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scn_);
    mjr_defaultContext(&con_);
    mjv_makeScene(model_, &scn_, 4000);
    mjr_makeContext(model_, &con_, mjFONTSCALE_150);
    cam_.distance = 2.0;
    cam_.azimuth = 90.0;
    cam_.elevation = -20.0;
  }

  ~MujocoWindow() {
    if (disabled_) return;
    mjr_freeContext(&con_);
    mjv_freeScene(&scn_);
    glfwDestroyWindow(window_);
    glfwTerminate();
  }

  bool running() const { return disabled_ || !glfwWindowShouldClose(window_); }

  void poll() {
    if (!disabled_) glfwPollEvents();
  }

  void render() {
    if (disabled_) return;
    cam_.lookat[0] = data_->qpos[0];
    cam_.lookat[1] = data_->qpos[1];
    cam_.lookat[2] = data_->qpos[2];
    mjrRect viewport{0, 0, 0, 0};
    glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
    mjv_updateScene(model_, data_, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
    mjr_render(viewport, &scn_, &con_);
    glfwSwapBuffers(window_);
  }

 private:
  mjModel* model_;
  mjData* data_;
  bool disabled_;
  GLFWwindow* window_ = nullptr;
  mjvCamera cam_{};
  mjvOption opt_{};
  mjvScene scn_{};
  mjrContext con_{};
};
#else
class MujocoWindow {
 public:
  MujocoWindow(mjModel*, mjData*, bool) {}
  bool running() const { return true; }
  void poll() {}
  void render() {}
};
#endif

}  // namespace

Sim2SimController::Sim2SimController(const fs::path& g1_deploy_dir,
                                     const fs::path& config_path,
                                     const std::string& model_name,
                                     PolicyKind kind)
    : g1_deploy_dir_(g1_deploy_dir), cfg_(load_config(config_path, g1_deploy_dir)), kind_(kind) {
  init_model(model_name);
}

Sim2SimController::~Sim2SimController() {
  if (data_) mj_deleteData(data_);
  if (model_) mj_deleteModel(model_);
}

void Sim2SimController::init_model(const std::string& model_name) {
  fs::path xml = resolve_asset_path(g1_deploy_dir_, cfg_.xml_path);
  fs::path policy_path = resolve_policy_path(g1_deploy_dir_, cfg_.policy_path, model_name);
  std::cout << "Loading MuJoCo model: " << xml << "\n";
  char error[1024] = {};
  model_ = mj_loadXML(xml.string().c_str(), nullptr, error, sizeof(error));
  if (!model_) throw std::runtime_error("Failed to load MuJoCo XML: " + xml.string() + " " + error);
  data_ = mj_makeData(model_);
  model_->opt.timestep = cfg_.sim_dt;
  policy_decimation_ = std::max(1, static_cast<int>(std::round(cfg_.control_dt / cfg_.sim_dt)));
  map_model_indices();

  std::cout << "Loading ONNX policy: " << policy_path << "\n";
  policy_ = std::make_unique<OnnxPolicy>(policy_path.string());
  configure_policy_state();

  for (int i = 0; i < num_joints_; ++i) data_->qpos[qpos_addr_[i]] = default_qpos_mj_[i];
  data_->qpos[2] = cfg_.init_height;
  mj_forward(model_, data_);
}

void Sim2SimController::load_policy_config(const fs::path& config_path,
                                           const std::string& model_name) {
  cfg_ = load_config(config_path, g1_deploy_dir_);
  fs::path policy_path = resolve_policy_path(g1_deploy_dir_, cfg_.policy_path, model_name);
  std::cout << "\n[PolicySwitch] Loading " << config_path << " / " << policy_path.filename().string() << "\n";
  policy_ = std::make_unique<OnnxPolicy>(policy_path.string());
  policy_decimation_ = std::max(1, static_cast<int>(std::round(cfg_.control_dt / cfg_.sim_dt)));
  model_->opt.timestep = cfg_.sim_dt;
  configure_policy_state();
}

void Sim2SimController::configure_policy_state() {
  kp_mj_ = reorder(cfg_.kps, cfg_.isaac_to_mujoco_map);
  kd_mj_ = reorder(cfg_.kds, cfg_.isaac_to_mujoco_map);
  default_qpos_mj_ = reorder(cfg_.default_joint_pos, cfg_.isaac_to_mujoco_map);
  target_qpos_mj_ = default_qpos_mj_;
  last_action_.assign(num_joints_, 0.0f);
  time_step_ = 0;
  ref_joint_pos_.clear();
  ref_joint_vel_.clear();
  ref_body_pos_w_.clear();
  ref_body_quat_w_.clear();
  obs_history_.clear();

  if (kind_ == PolicyKind::Walk || kind_ == PolicyKind::Amp) {
    int frame_dim = 9 + 3 * num_joints_;
    obs_history_.assign(static_cast<size_t>(std::max(1, cfg_.history_length)), std::vector<float>(frame_dim, 0.0f));
  }
  if (kind_ == PolicyKind::Mimic && cfg_.policy_type == "flat") {
    int frame_dim = 9 + 3 * num_joints_;
    obs_history_.assign(static_cast<size_t>(std::max(1, cfg_.history_length)), std::vector<float>(frame_dim, 0.0f));
  }

  if (kind_ == PolicyKind::Attention) {
    terrain_body_id_ = mj_name2id(model_, mjOBJ_BODY, cfg_.terrain_sensor_body.c_str());
    if (terrain_body_id_ < 0) throw std::runtime_error("Terrain sensor body not found: " + cfg_.terrain_sensor_body);
    scan_offset_local_ = {cfg_.terrain_sensor_offset[0], cfg_.terrain_sensor_offset[1], cfg_.terrain_sensor_offset[2]};
    int length = cfg_.terrain_map_length;
    int width = cfg_.terrain_map_width;
    scan_grid_local_.reserve(static_cast<size_t>(length * width));
    for (int iy = 0; iy < width; ++iy) {
      for (int ix = 0; ix < length; ++ix) {
        double x = -0.5 * cfg_.terrain_map_size[0] + cfg_.terrain_map_size[0] * ix / std::max(1, length - 1);
        double y = -0.5 * cfg_.terrain_map_size[1] + cfg_.terrain_map_size[1] * iy / std::max(1, width - 1);
        scan_grid_local_.push_back({x, y, 0.0});
      }
    }
    ray_geomgroup_.assign(mjNGROUP, 0);
    ray_geomgroup_[0] = 1;
    for (int i = 0; i < model_->ngeom; ++i) {
      if (model_->geom_bodyid[i] != 0) model_->geom_group[i] = 1;
    }
  }

  if (kind_ == PolicyKind::Mimic && cfg_.policy_type == "tracking") {
    anchor_body_id_ = mj_name2id(model_, mjOBJ_BODY, cfg_.anchor_body_name.c_str());
    auto it = std::find(cfg_.body_names.begin(), cfg_.body_names.end(), cfg_.anchor_body_name);
    if (anchor_body_id_ < 0 || it == cfg_.body_names.end()) {
      throw std::runtime_error("Mimic anchor body not found in MuJoCo/config body_names: " + cfg_.anchor_body_name);
    }
    anchor_body_idx_ = static_cast<int>(std::distance(cfg_.body_names.begin(), it));
    auto outs = policy_->run_named(std::vector<float>(static_cast<size_t>(cfg_.num_obs), 0.0f), 0.0f);
    ref_joint_pos_ = tensor_to_vector(outs.at(1));
    ref_joint_vel_ = tensor_to_vector(outs.at(2));
    ref_body_pos_w_ = tensor_to_vector(outs.at(3));
    ref_body_quat_w_ = tensor_to_vector(outs.at(4));
  }
}

void Sim2SimController::map_model_indices() {
  num_joints_ = static_cast<int>(cfg_.joint_names_mujoco.size());
  qpos_addr_.resize(num_joints_);
  qvel_addr_.resize(num_joints_);
  actuator_id_.resize(num_joints_);
  for (int i = 0; i < num_joints_; ++i) {
    int jid = mj_name2id(model_, mjOBJ_JOINT, cfg_.joint_names_mujoco[i].c_str());
    int aid = mj_name2id(model_, mjOBJ_ACTUATOR, cfg_.actuator_names_mujoco[i].c_str());
    if (jid < 0) throw std::runtime_error("Joint not found: " + cfg_.joint_names_mujoco[i]);
    if (aid < 0) throw std::runtime_error("Actuator not found: " + cfg_.actuator_names_mujoco[i]);
    qpos_addr_[i] = model_->jnt_qposadr[jid];
    qvel_addr_[i] = model_->jnt_dofadr[jid];
    actuator_id_[i] = aid;
    if (model_->actuator_trnid[2 * aid] != jid) {
      throw std::runtime_error("Actuator does not drive expected joint: " + cfg_.actuator_names_mujoco[i]);
    }
  }
}

void Sim2SimController::validate() const {
  auto check = [&](const char* name, size_t n) {
    if (n != static_cast<size_t>(num_joints_)) {
      throw std::runtime_error(std::string(name) + " length mismatch.");
    }
  };
  check("kps", cfg_.kps.size());
  check("kds", cfg_.kds.size());
  check("default_joint_pos", cfg_.default_joint_pos.size());
  check("action_scale", cfg_.action_scale.size());
  check("mujoco_to_isaac_map", cfg_.mujoco_to_isaac_map.size());
  check("isaac_to_mujoco_map", cfg_.isaac_to_mujoco_map.size());
  if (cfg_.num_actions != num_joints_) throw std::runtime_error("num_actions does not match joint count.");
  if (policy_->input_dim() > 0 && policy_->input_dim() != cfg_.num_obs) {
    throw std::runtime_error("ONNX input dim does not match YAML num_obs.");
  }
  if (policy_->output_dim() > 0 && policy_->output_dim() != cfg_.num_actions) {
    throw std::runtime_error("ONNX output dim does not match YAML num_actions.");
  }
  std::cout << "[Check] joints=" << num_joints_ << " num_obs=" << cfg_.num_obs
            << " onnx_input=" << policy_->input_dim() << " onnx_output=" << policy_->output_dim() << "\n";
}

std::vector<float> Sim2SimController::qpos_mujoco_order() const {
  std::vector<float> out(num_joints_);
  for (int i = 0; i < num_joints_; ++i) out[i] = static_cast<float>(data_->qpos[qpos_addr_[i]]);
  return out;
}

std::vector<float> Sim2SimController::qvel_mujoco_order() const {
  std::vector<float> out(num_joints_);
  for (int i = 0; i < num_joints_; ++i) out[i] = static_cast<float>(data_->qvel[qvel_addr_[i]]);
  return out;
}

void Sim2SimController::pd_control() {
  for (int i = 0; i < num_joints_; ++i) {
    double q = data_->qpos[qpos_addr_[i]];
    double dq = data_->qvel[qvel_addr_[i]];
    data_->ctrl[actuator_id_[i]] = kp_mj_[i] * (target_qpos_mj_[i] - q) + kd_mj_[i] * (0.0 - dq);
  }
}

std::vector<float> Sim2SimController::build_walk_like_obs(const VelocityCommand& command) {
  QuatXyzw quat = wxyz_to_xyzw(data_->qpos + 3);
  Vec3 proj_grav = compute_projected_gravity(quat);
  Vec3 base_ang{static_cast<float>(data_->qvel[3]), static_cast<float>(data_->qvel[4]), static_cast<float>(data_->qvel[5])};
  auto q_mj = qpos_mujoco_order();
  auto dq_mj = qvel_mujoco_order();
  auto q_isaac = reorder(q_mj, cfg_.mujoco_to_isaac_map);
  auto dq_isaac = reorder(dq_mj, cfg_.mujoco_to_isaac_map);
  auto q_rel = subtract(q_isaac, cfg_.default_joint_pos);
  return build_proprio_obs(base_ang, proj_grav, command, q_rel, dq_isaac, last_action_,
                           cfg_.ang_vel_scale, cfg_.dof_pos_scale, cfg_.dof_vel_scale,
                           cfg_.last_action_scale, cfg_.command_scale);
}

std::vector<float> Sim2SimController::compute_terrain_scan() {
  if (terrain_body_id_ < 0) return {};
  mj_kinematics(model_, data_);
  const double* body_pos = data_->xpos + 3 * terrain_body_id_;
  const double* xmat = data_->xmat + 9 * terrain_body_id_;
  double yaw = yaw_from_xmat(xmat);
  double c = std::cos(yaw), s = std::sin(yaw);

  std::vector<float> terrain;
  terrain.reserve(scan_grid_local_.size() * 3);
  for (const auto& p : scan_grid_local_) {
    double start[3] = {
        body_pos[0] + scan_offset_local_[0] + c * p[0] - s * p[1],
        body_pos[1] + scan_offset_local_[1] + s * p[0] + c * p[1],
        body_pos[2] + scan_offset_local_[2],
    };
    double dir[3] = {0.0, 0.0, -1.0};
    int geomid = -1;
    double dist = mj_ray(model_, data_, start, dir,
                         reinterpret_cast<const mjtByte*>(ray_geomgroup_.data()),
                         static_cast<mjtByte>(cfg_.terrain_flg_static),
                         cfg_.terrain_bodyexclude, &geomid);
    double hit_z = (dist >= 0.0 && std::isfinite(dist)) ? start[2] - dist : body_pos[2];
    double rel_x = start[0] - body_pos[0];
    double rel_y = start[1] - body_pos[1];
    double local_x = c * rel_x + s * rel_y;
    double local_y = -s * rel_x + c * rel_y;
    double local_z = std::clamp(hit_z - body_pos[2],
                                static_cast<double>(cfg_.terrain_map_z_clip[0]),
                                static_cast<double>(cfg_.terrain_map_z_clip[1]));
    terrain.push_back(static_cast<float>(local_x));
    terrain.push_back(static_cast<float>(local_y));
    terrain.push_back(static_cast<float>(local_z));
  }
  return terrain;
}

std::vector<float> Sim2SimController::build_attention_obs(const VelocityCommand& command) {
  auto proprio = build_walk_like_obs(command);
  auto terrain = compute_terrain_scan();
  proprio.insert(proprio.end(), terrain.begin(), terrain.end());
  return proprio;
}

std::vector<float> Sim2SimController::build_mimic_obs() {
  if (cfg_.policy_type == "flat") {
    auto frame = build_walk_like_obs({});
    obs_history_.pop_front();
    obs_history_.push_back(frame);
    std::vector<std::vector<float>> hist(obs_history_.begin(), obs_history_.end());
    return stack_history_group_major(hist, num_joints_);
  }

  auto q_mj = qpos_mujoco_order();
  auto dq_mj = qvel_mujoco_order();
  auto q_isaac = reorder(q_mj, cfg_.mujoco_to_isaac_map);
  auto dq_isaac = reorder(dq_mj, cfg_.mujoco_to_isaac_map);
  auto q_rel = subtract(q_isaac, cfg_.default_joint_pos);

  QuatWxyz base_q{static_cast<float>(data_->qpos[3]), static_cast<float>(data_->qpos[4]),
                  static_cast<float>(data_->qpos[5]), static_cast<float>(data_->qpos[6])};
  Vec3 base_lin_w{static_cast<float>(data_->qvel[0]), static_cast<float>(data_->qvel[1]), static_cast<float>(data_->qvel[2])};
  Vec3 base_lin_b = quat_apply_wxyz(quat_inv_wxyz(base_q), base_lin_w);
  Vec3 base_ang_b{static_cast<float>(data_->qvel[3]), static_cast<float>(data_->qvel[4]), static_cast<float>(data_->qvel[5])};

  const double* robot_pos_d = data_->xpos + 3 * anchor_body_id_;
  const double* robot_quat_d = data_->xquat + 4 * anchor_body_id_;
  Vec3 robot_pos{static_cast<float>(robot_pos_d[0]), static_cast<float>(robot_pos_d[1]), static_cast<float>(robot_pos_d[2])};
  QuatWxyz robot_q{static_cast<float>(robot_quat_d[0]), static_cast<float>(robot_quat_d[1]),
                   static_cast<float>(robot_quat_d[2]), static_cast<float>(robot_quat_d[3])};
  size_t pos_i = static_cast<size_t>(anchor_body_idx_ * 3);
  size_t quat_i = static_cast<size_t>(anchor_body_idx_ * 4);
  Vec3 motion_pos{ref_body_pos_w_.at(pos_i), ref_body_pos_w_.at(pos_i + 1), ref_body_pos_w_.at(pos_i + 2)};
  QuatWxyz motion_q{ref_body_quat_w_.at(quat_i), ref_body_quat_w_.at(quat_i + 1),
                    ref_body_quat_w_.at(quat_i + 2), ref_body_quat_w_.at(quat_i + 3)};
  QuatWxyz inv_robot = quat_inv_wxyz(robot_q);
  Vec3 delta{motion_pos[0] - robot_pos[0], motion_pos[1] - robot_pos[1], motion_pos[2] - robot_pos[2]};
  Vec3 anchor_pos_b = quat_apply_wxyz(inv_robot, delta);
  QuatWxyz anchor_quat_b = quat_mul_wxyz(inv_robot, motion_q);
  auto mat = matrix_from_quat_wxyz(anchor_quat_b);

  std::vector<float> obs;
  append(obs, ref_joint_pos_);
  append(obs, ref_joint_vel_);
  if (cfg_.include_state_estimation) append3(obs, anchor_pos_b);
  obs.insert(obs.end(), {mat[0], mat[1], mat[3], mat[4], mat[6], mat[7]});
  if (cfg_.include_state_estimation) append3(obs, base_lin_b);
  append3(obs, base_ang_b);
  append(obs, q_rel);
  append(obs, dq_isaac);
  append(obs, last_action_);
  return obs;
}

void Sim2SimController::apply_action(const std::vector<float>& raw_action) {
  std::vector<float> action = raw_action;
  action.resize(static_cast<size_t>(num_joints_), 0.0f);
  if (cfg_.has_action_clip) {
    for (float& v : action) v = std::clamp(v, -cfg_.action_clip, cfg_.action_clip);
  }
  last_action_ = action;
  std::vector<float> target_isaac(num_joints_);
  for (int i = 0; i < num_joints_; ++i) {
    target_isaac[i] = action[i] * cfg_.action_scale.at(static_cast<size_t>(i)) + cfg_.default_joint_pos.at(static_cast<size_t>(i));
  }
  target_qpos_mj_ = reorder(target_isaac, cfg_.isaac_to_mujoco_map);
}

void Sim2SimController::policy_step(CommandController& input) {
  VelocityCommand cmd = input.velocity();
  std::vector<float> obs_input;
  if (kind_ == PolicyKind::Attention) {
    obs_input = build_attention_obs(cmd);
  } else if (kind_ == PolicyKind::Mimic) {
    obs_input = build_mimic_obs();
  } else {
    auto frame = build_walk_like_obs(cmd);
    obs_history_.pop_front();
    obs_history_.push_back(frame);
    std::vector<std::vector<float>> hist(obs_history_.begin(), obs_history_.end());
    obs_input = stack_history_group_major(hist, num_joints_);
  }

  std::vector<float> action;
  if (kind_ == PolicyKind::Mimic && cfg_.policy_type == "tracking") {
    auto outs = policy_->run_named(obs_input, static_cast<float>(time_step_));
    action = tensor_to_vector(outs.at(0));
    ref_joint_pos_ = tensor_to_vector(outs.at(1));
    ref_joint_vel_ = tensor_to_vector(outs.at(2));
    ref_body_pos_w_ = tensor_to_vector(outs.at(3));
    ref_body_quat_w_ = tensor_to_vector(outs.at(4));
    ++time_step_;
    if (cfg_.motion_total_steps > 0 && time_step_ >= cfg_.motion_total_steps) time_step_ = 0;
  } else {
    action = policy_->run_single(obs_input);
  }
  apply_action(action);
}

void Sim2SimController::run(CommandController& input, const RunOptions& options) {
  validate();
  if (options.check) {
    if (kind_ == PolicyKind::Attention) {
      auto scan = compute_terrain_scan();
      std::cout << "[Check] attention terrain scan values=" << scan.size() << "\n";
    }
    return;
  }

  bool headless = options.no_render;
#ifndef G1_CPP_HAS_GLFW
  headless = true;
  if (!options.no_render) std::cout << "[viewer] GLFW support is not compiled; running headless.\n";
#endif
  MujocoWindow window(model_, data_, headless);
  int motiontime = 0;
  int active_policy = 1;
  int render_decimation = std::max(1, static_cast<int>(std::round(0.02 / cfg_.sim_dt)));
  double start = wall_time();
  input.start();

  try {
    while (window.running() && !input.exit_requested()) {
      if (kind_ == PolicyKind::Mimic) {
        int requested = input.active_policy();
        if (requested != 0 && requested != active_policy) {
          if (requested == 1) {
            load_policy_config(resolve_config_path(g1_deploy_dir_, "g1_walk.yaml"), "g1_flat_1.onnx");
            active_policy = requested;
          } else if (requested == 2) {
            load_policy_config(resolve_config_path(g1_deploy_dir_, options.config_name), options.model_name);
            active_policy = requested;
          } else if (requested == 3) {
            load_policy_config(resolve_config_path(g1_deploy_dir_, "g1_mimic.yaml"), "g1_jump.onnx");
            active_policy = requested;
          } else if (requested == 4) {
            load_policy_config(resolve_config_path(g1_deploy_dir_, "g1_mimic.yaml"), "g1_dance.onnx");
            active_policy = requested;
          }
          render_decimation = std::max(1, static_cast<int>(std::round(0.02 / cfg_.sim_dt)));
        }
      }

      std::fill(data_->xfrc_applied, data_->xfrc_applied + 6 * model_->nbody, 0.0);
      pd_control();
      mj_step(model_, data_);
      ++motiontime;

      if (motiontime % policy_decimation_ == 0) {
        policy_step(input);
      }

      window.poll();
      if (motiontime % render_decimation == 0) {
        window.render();
      }

      double expected = start + motiontime * cfg_.sim_dt;
      double sleep_s = expected - wall_time();
      if (sleep_s > 0.0) std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));

      int print_period = std::max(1, static_cast<int>(1.0 / cfg_.sim_dt));
      if (motiontime % print_period == 0) {
        VelocityCommand v = input.velocity();
        double elapsed = wall_time() - start;
        double hz = elapsed > 0.0 ? motiontime / elapsed : 0.0;
        std::cout << "[Sim] t=" << motiontime * cfg_.sim_dt
                  << "s height=" << data_->qpos[2]
                  << " cmd=[" << v.vx << ", " << v.vy << ", " << v.vyaw << "]"
                  << " hz=" << hz << "\n";
      }
    }
  } catch (...) {
    input.stop();
    throw;
  }
  input.stop();
}

std::unique_ptr<CommandController> make_controller(const Config& cfg, const RunOptions& options) {
  if (options.input == "const") {
    auto vx = cfg_range(cfg, "lin_vel_x", {-1.0f, 1.0f});
    auto vy = cfg_range(cfg, "lin_vel_y", {-0.5f, 0.5f});
    auto vyaw = cfg_range(cfg, "ang_vel_z", {-1.0f, 1.0f});
    float cvx = std::clamp(options.const_vx, vx.lo, vx.hi);
    float cvy = std::clamp(options.const_vy, vy.lo, vy.hi);
    float cvyaw = std::clamp(options.const_vyaw, vyaw.lo, vyaw.hi);
    return std::make_unique<ConstantController>(cvx, cvy, cvyaw, options.const_warmup, 1.0f);
  }
  if (options.input == "gamepad") {
    std::cout << "[input] Native C++ gamepad mapping is not implemented yet; using keyboard controls.\n";
  }
  return std::make_unique<KeyboardController>(
      cfg_range(cfg, "lin_vel_x", {-1.0f, 1.0f}),
      cfg_range(cfg, "lin_vel_y", {-0.5f, 0.5f}),
      cfg_range(cfg, "ang_vel_z", {-1.0f, 1.0f}));
}

int run_sim2sim_main(int argc, char** argv, PolicyKind kind, const char* default_config, const char* default_model) {
  RunOptions options;
  options.kind = kind;
  options.config_name = default_config;
  options.model_name = default_model;

  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    auto need_value = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
      return argv[++i];
    };
    if (a == "--config") options.config_name = need_value(a);
    else if (a == "--model") options.model_name = need_value(a);
    else if (a == "--input") options.input = need_value(a);
    else if (a == "--check") options.check = true;
    else if (a == "--no_render") options.no_render = true;
    else if (a == "--debug_policy") options.debug_policy = true;
    else if (a == "--show_rays") options.show_rays = true;
    else if (a == "--const_vx") options.const_vx = std::stof(need_value(a));
    else if (a == "--const_vy") options.const_vy = std::stof(need_value(a));
    else if (a == "--const_vyaw") options.const_vyaw = std::stof(need_value(a));
    else if (a == "--const_warmup") options.const_warmup = std::stof(need_value(a));
    else if (a == "-h" || a == "--help") {
      std::cout << "Usage: " << argv[0] << " [--config YAML] [--model ONNX] [--input keyboard|const|gamepad]\n"
                << "       [--check] [--no_render] [--const_vx V] [--const_vy V] [--const_vyaw W]\n";
      return 0;
    }
  }

  (void)argv;
#ifdef G1_CPP_SOURCE_DIR
  fs::path g1_cpp_dir = G1_CPP_SOURCE_DIR;
#else
  fs::path g1_cpp_dir = fs::current_path();
#endif
  if (fs::exists(fs::current_path() / "deploy/g1_deploy")) {
    g1_cpp_dir = fs::current_path() / "deploy/g1_deploy/g1_cpp";
  } else if (fs::exists(fs::current_path() / "g1_cpp")) {
    g1_cpp_dir = fs::current_path() / "g1_cpp";
  }
  fs::path g1_deploy_dir = fs::weakly_canonical(g1_cpp_dir / "..");
  fs::path config_path = resolve_config_path(g1_deploy_dir, options.config_name);
  fs::path startup_config_path = config_path;
  std::string startup_model_name = options.model_name;
  if (kind == PolicyKind::Mimic) {
    startup_config_path = resolve_config_path(g1_deploy_dir, "g1_walk.yaml");
    startup_model_name = "g1_flat_1.onnx";
  }

  try {
    Config input_cfg = load_config(startup_config_path, g1_deploy_dir);
    auto input = make_controller(input_cfg, options);
    Sim2SimController controller(g1_deploy_dir, startup_config_path, startup_model_name, kind);
    controller.run(*input, options);
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << "\n";
    return 1;
  }
  return 0;
}

}  // namespace g1
