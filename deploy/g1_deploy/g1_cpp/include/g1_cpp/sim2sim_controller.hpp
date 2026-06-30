#pragma once

#include "g1_cpp/config.hpp"
#include "g1_cpp/controllers.hpp"
#include "g1_cpp/onnx_policy.hpp"

#include <mujoco/mujoco.h>

#include <array>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace g1 {

enum class PolicyKind {
  Walk,
  Amp,
  Mimic,
  Attention,
};

struct RunOptions {
  PolicyKind kind = PolicyKind::Walk;
  std::string config_name;
  std::string model_name;
  std::string input = "keyboard";
  bool check = false;
  bool no_render = false;
  bool debug_policy = false;
  bool show_rays = false;
  float const_vx = 0.0f;
  float const_vy = 0.0f;
  float const_vyaw = 0.0f;
  float const_warmup = 2.0f;
};

class Sim2SimController {
 public:
  Sim2SimController(const std::filesystem::path& g1_deploy_dir,
                    const std::filesystem::path& config_path,
                    const std::string& model_name,
                    PolicyKind kind);
  ~Sim2SimController();

  void validate() const;
  void run(CommandController& input, const RunOptions& options);
  std::vector<float> compute_terrain_scan();

 private:
  void init_model(const std::string& model_name);
  void load_policy_config(const std::filesystem::path& config_path,
                          const std::string& model_name);
  void configure_policy_state();
  void map_model_indices();
  void pd_control();
  void policy_step(CommandController& input);
  std::vector<float> build_walk_like_obs(const VelocityCommand& command);
  std::vector<float> build_attention_obs(const VelocityCommand& command);
  std::vector<float> build_mimic_obs();
  void apply_action(const std::vector<float>& action);
  std::vector<float> qpos_mujoco_order() const;
  std::vector<float> qvel_mujoco_order() const;

  std::filesystem::path g1_deploy_dir_;
  Config cfg_;
  PolicyKind kind_;
  std::unique_ptr<OnnxPolicy> policy_;

  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  int policy_decimation_ = 4;
  int num_joints_ = 0;
  int terrain_body_id_ = -1;
  int anchor_body_id_ = -1;
  int anchor_body_idx_ = -1;
  int time_step_ = 0;

  std::vector<int> qpos_addr_;
  std::vector<int> qvel_addr_;
  std::vector<int> actuator_id_;
  std::vector<float> kp_mj_;
  std::vector<float> kd_mj_;
  std::vector<float> default_qpos_mj_;
  std::vector<float> target_qpos_mj_;
  std::vector<float> last_action_;
  std::deque<std::vector<float>> obs_history_;
  std::vector<float> ref_joint_pos_;
  std::vector<float> ref_joint_vel_;
  std::vector<float> ref_body_pos_w_;
  std::vector<float> ref_body_quat_w_;

  std::vector<std::array<double, 3>> scan_grid_local_;
  std::array<double, 3> scan_offset_local_{0.0, 0.0, 20.0};
  std::vector<unsigned char> ray_geomgroup_;
};

std::unique_ptr<CommandController> make_controller(const Config& cfg, const RunOptions& options);
int run_sim2sim_main(int argc, char** argv, PolicyKind kind, const char* default_config, const char* default_model);

}  // namespace g1
