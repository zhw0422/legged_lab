#include "g1_cpp/config.hpp"

#include <mujoco/mujoco.h>

#ifdef G1_CPP_HAS_GLFW
#include <GLFW/glfw3.h>
#endif

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <linux/joystick.h>
#include <termios.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

using LowCmdHg = unitree_hg::msg::dds_::LowCmd_;
using LowStateHg = unitree_hg::msg::dds_::LowState_;
using SportModeState = unitree_go::msg::dds_::SportModeState_;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelSubscriber;

#pragma pack(push, 1)
struct RemoteButtons {
  uint8_t r1 : 1;
  uint8_t l1 : 1;
  uint8_t start : 1;
  uint8_t select : 1;
  uint8_t r2 : 1;
  uint8_t l2 : 1;
  uint8_t f1 : 1;
  uint8_t f2 : 1;
  uint8_t a : 1;
  uint8_t b : 1;
  uint8_t x : 1;
  uint8_t y : 1;
  uint8_t up : 1;
  uint8_t right : 1;
  uint8_t down : 1;
  uint8_t left : 1;
};

struct RemotePacket {
  uint8_t head[2] = {0xFE, 0xEF};
  RemoteButtons btn{};
  float lx = 0.0f;
  float rx = 0.0f;
  float ry = 0.0f;
  float l2 = 0.0f;
  float ly = 0.0f;
  uint8_t idle[16] = {};
};
#pragma pack(pop)

static_assert(sizeof(RemotePacket) == 40, "Unitree wireless remote packet must be 40 bytes.");

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t crc = 0xFFFFFFFF;
  constexpr uint32_t poly = 0x04c11db7;
  for (uint32_t i = 0; i < len; ++i) {
    xbit = 1u << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; ++bits) {
      if (crc & 0x80000000) {
        crc <<= 1;
        crc ^= poly;
      } else {
        crc <<= 1;
      }
      if (data & xbit) crc ^= poly;
      xbit >>= 1;
    }
  }
  return crc;
}

double wall_time() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

fs::path deploy_dir() {
  return fs::path(G1_CPP_SOURCE_DIR).parent_path();
}

fs::path resolve_config_path(const fs::path& g1_dir, const std::string& name) {
  fs::path p(name);
  if (p.is_absolute() || fs::exists(p)) return p;
  if (!p.empty() && *p.begin() == "config") return g1_dir / p;
  return g1_dir / "config" / p;
}

fs::path resolve_asset_path(const fs::path& g1_dir, const std::string& xml_path) {
  fs::path p(xml_path);
  if (p.is_absolute()) return p;
  return g1_dir / p;
}

std::array<float, 3> quat_to_rpy(const double* q) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double sinr_cosp = 2.0 * (w * x + y * z);
  double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  double roll = std::atan2(sinr_cosp, cosr_cosp);
  double sinp = 2.0 * (w * y - z * x);
  double pitch = std::abs(sinp) >= 1.0 ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);
  double siny_cosp = 2.0 * (w * z + x * y);
  double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  double yaw = std::atan2(siny_cosp, cosy_cosp);
  return {static_cast<float>(roll), static_cast<float>(pitch), static_cast<float>(yaw)};
}

class TerminalRemote {
 public:
  TerminalRemote() {
    if (isatty(STDIN_FILENO)) {
      valid_ = true;
      tcgetattr(STDIN_FILENO, &old_);
      termios raw = old_;
      raw.c_lflag &= static_cast<unsigned>(~(ICANON | ECHO));
      tcsetattr(STDIN_FILENO, TCSANOW, &raw);
      old_flags_ = fcntl(STDIN_FILENO, F_GETFL, 0);
      fcntl(STDIN_FILENO, F_SETFL, old_flags_ | O_NONBLOCK);
    }
  }

  ~TerminalRemote() {
    if (valid_) {
      tcsetattr(STDIN_FILENO, TCSANOW, &old_);
      fcntl(STDIN_FILENO, F_SETFL, old_flags_);
    }
  }

  void poll() {
    clear_pulses();
    if (!valid_) return;
    char ch = 0;
    while (read(STDIN_FILENO, &ch, 1) == 1) {
      switch (ch) {
        case '\n':
        case '\r':
        case '1':
          pulse_start();
          break;
        case '2':
          pulse_a();
          break;
        case '3':
          pulse_b();
          break;
        case '4':
          pulse_x();
          break;
        case '5':
          pulse_y();
          break;
        case '9':
          request_exit();
          break;
        case 'w':
          adjust_ly(0.1f);
          break;
        case 's':
          adjust_ly(-0.1f);
          break;
        case 'a':
          adjust_lx(0.1f);
          break;
        case 'd':
          adjust_lx(-0.1f);
          break;
        case 'q':
          adjust_rx(0.1f);
          break;
        case 'e':
          adjust_rx(-0.1f);
          break;
        case ' ':
        case '0':
          zero_axes();
          break;
        case 3:
        case 27:
          request_exit();
          break;
        default:
          break;
      }
    }
  }

  const RemotePacket& packet() const { return packet_; }
  bool exit_requested() const { return exit_requested_; }
  uint16_t key_mask() const {
    uint16_t mask = 0;
    mask |= static_cast<uint16_t>(packet_.btn.r1) << 0;
    mask |= static_cast<uint16_t>(packet_.btn.l1) << 1;
    mask |= static_cast<uint16_t>(packet_.btn.start) << 2;
    mask |= static_cast<uint16_t>(packet_.btn.select) << 3;
    mask |= static_cast<uint16_t>(packet_.btn.r2) << 4;
    mask |= static_cast<uint16_t>(packet_.btn.l2) << 5;
    mask |= static_cast<uint16_t>(packet_.btn.f1) << 6;
    mask |= static_cast<uint16_t>(packet_.btn.f2) << 7;
    mask |= static_cast<uint16_t>(packet_.btn.a) << 8;
    mask |= static_cast<uint16_t>(packet_.btn.b) << 9;
    mask |= static_cast<uint16_t>(packet_.btn.x) << 10;
    mask |= static_cast<uint16_t>(packet_.btn.y) << 11;
    mask |= static_cast<uint16_t>(packet_.btn.up) << 12;
    mask |= static_cast<uint16_t>(packet_.btn.right) << 13;
    mask |= static_cast<uint16_t>(packet_.btn.down) << 14;
    mask |= static_cast<uint16_t>(packet_.btn.left) << 15;
    return mask;
  }

  void pulse_start() { held_start_until_ = wall_time() + button_hold_s_; }
  void pulse_a() { held_a_until_ = wall_time() + button_hold_s_; }
  void pulse_b() { held_b_until_ = wall_time() + button_hold_s_; }
  void pulse_x() { held_x_until_ = wall_time() + button_hold_s_; }
  void pulse_y() { held_y_until_ = wall_time() + button_hold_s_; }
  void request_exit() {
    held_select_until_ = wall_time() + button_hold_s_;
    exit_requested_ = true;
  }
  void adjust_lx(float delta) { packet_.lx = std::clamp(packet_.lx + delta, -1.0f, 1.0f); }
  void adjust_ly(float delta) { packet_.ly = std::clamp(packet_.ly + delta, -1.0f, 1.0f); }
  void adjust_rx(float delta) { packet_.rx = std::clamp(packet_.rx + delta, -1.0f, 1.0f); }
  void set_lx(float value) { packet_.lx = std::clamp(value, -1.0f, 1.0f); }
  void set_ly(float value) { packet_.ly = std::clamp(value, -1.0f, 1.0f); }
  void set_rx(float value) { packet_.rx = std::clamp(value, -1.0f, 1.0f); }
  void set_ry(float value) { packet_.ry = std::clamp(value, -1.0f, 1.0f); }
  void zero_axes() { packet_.lx = packet_.ly = packet_.rx = packet_.ry = 0.0f; }

 private:
  void clear_pulses() {
    double now = wall_time();
    packet_.btn.start = held_start_until_ >= now;
    packet_.btn.select = held_select_until_ >= now;
    packet_.btn.a = held_a_until_ >= now;
    packet_.btn.b = held_b_until_ >= now;
    packet_.btn.x = held_x_until_ >= now;
    packet_.btn.y = held_y_until_ >= now;
  }

  bool valid_ = false;
  bool exit_requested_ = false;
  int old_flags_ = 0;
  termios old_{};
  RemotePacket packet_{};
  double held_start_until_ = 0.0;
  double held_select_until_ = 0.0;
  double held_a_until_ = 0.0;
  double held_b_until_ = 0.0;
  double held_x_until_ = 0.0;
  double held_y_until_ = 0.0;
  double button_hold_s_ = 0.2;
};

struct JoystickMap {
  int axis_lx = 0;
  int axis_ly = 1;
  int axis_rx = 2;
  int axis_ry = 3;
  int button_a = 0;
  int button_b = 1;
  int button_x = 3;
  int button_y = 4;
  int button_select = 10;
  int button_start = 11;
};

JoystickMap joystick_map_for(const std::string& type) {
  if (type == "xbox") {
    return JoystickMap{0, 1, 3, 4, 0, 1, 2, 3, 6, 7};
  }
  return JoystickMap{0, 1, 2, 3, 0, 1, 3, 4, 10, 11};
}

class LinuxJoystick {
 public:
  LinuxJoystick(bool enabled, std::string type, TerminalRemote* remote)
      : enabled_(enabled), map_(joystick_map_for(type)), remote_(remote) {
    if (!enabled_) return;
    const char* env_path = std::getenv("G1_CPP_JOYSTICK");
    std::vector<std::string> candidates;
    if (env_path && env_path[0] != '\0') candidates.emplace_back(env_path);
    candidates.emplace_back("/dev/input/js0");
    candidates.emplace_back("/dev/input/js1");
    for (const auto& path : candidates) {
      fd_ = open(path.c_str(), O_RDONLY | O_NONBLOCK);
      if (fd_ >= 0) {
        path_ = path;
        std::cout << "[SDK2 Bridge/CPP] gamepad=" << path_ << " type=" << type << "\n";
        break;
      }
    }
    if (fd_ < 0) {
      std::cout << "[SDK2 Bridge/CPP] no /dev/input/js* gamepad found; use viewer or bridge terminal keys.\n";
    }
  }

  ~LinuxJoystick() {
    if (fd_ >= 0) close(fd_);
  }

  void poll() {
    if (fd_ < 0 || !remote_) return;
    js_event e{};
    while (read(fd_, &e, sizeof(e)) == static_cast<ssize_t>(sizeof(e))) {
      e.type &= ~JS_EVENT_INIT;
      if (e.type == JS_EVENT_AXIS) {
        axes_[e.number] = normalize_axis(e.value);
      } else if (e.type == JS_EVENT_BUTTON) {
        buttons_[e.number] = e.value != 0;
        if (e.value != 0) {
          std::cout << "[SDK2 Bridge/CPP] gamepad button " << static_cast<int>(e.number) << " pressed\n";
        }
      }
    }
    apply();
  }

 private:
  float normalize_axis(int16_t value) const {
    float v = static_cast<float>(value) / 32767.0f;
    return std::fabs(v) < 0.05f ? 0.0f : std::clamp(v, -1.0f, 1.0f);
  }

  bool button(int index) const {
    auto it = buttons_.find(static_cast<uint8_t>(index));
    return it != buttons_.end() && it->second;
  }

  float axis(int index) const {
    auto it = axes_.find(static_cast<uint8_t>(index));
    return it == axes_.end() ? 0.0f : it->second;
  }

  void apply() {
    remote_->set_lx(axis(map_.axis_lx));
    remote_->set_ly(-axis(map_.axis_ly));
    remote_->set_rx(axis(map_.axis_rx));
    remote_->set_ry(-axis(map_.axis_ry));
    if (button(map_.button_start)) remote_->pulse_start();
    if (button(map_.button_select)) remote_->request_exit();
    if (button(map_.button_a)) remote_->pulse_a();
    if (button(map_.button_b)) remote_->pulse_b();
    if (button(map_.button_x)) remote_->pulse_x();
    if (button(map_.button_y)) remote_->pulse_y();
  }

  bool enabled_ = false;
  int fd_ = -1;
  std::string path_;
  JoystickMap map_;
  TerminalRemote* remote_ = nullptr;
  std::unordered_map<uint8_t, float> axes_;
  std::unordered_map<uint8_t, bool> buttons_;
};

#ifdef G1_CPP_HAS_GLFW
class MujocoWindow {
 public:
  MujocoWindow(mjModel* model,
               mjData* data,
               bool disabled,
               TerminalRemote* remote,
               bool* elastic_enabled,
               double* elastic_length)
      : model_(model),
        data_(data),
        disabled_(disabled),
        remote_(remote),
        elastic_enabled_(elastic_enabled),
        elastic_length_(elastic_length) {
    if (disabled_) return;
    if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW.");
    window_ = glfwCreateWindow(1280, 900, "G1 C++ SDK2 MuJoCo Bridge", nullptr, nullptr);
    if (!window_) throw std::runtime_error("Failed to create GLFW window.");
    glfwMakeContextCurrent(window_);
    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, &MujocoWindow::key_callback);
    glfwSetCursorPosCallback(window_, &MujocoWindow::cursor_pos_callback);
    glfwSetMouseButtonCallback(window_, &MujocoWindow::mouse_button_callback);
    glfwSetScrollCallback(window_, &MujocoWindow::scroll_callback);
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
  static void key_callback(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    auto* self = static_cast<MujocoWindow*>(glfwGetWindowUserPointer(window));
    if (!self) return;
    self->handle_key(key);
  }

  static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    auto* self = static_cast<MujocoWindow*>(glfwGetWindowUserPointer(window));
    if (!self) return;
    self->handle_cursor(xpos, ypos);
  }

  static void mouse_button_callback(GLFWwindow* window, int button, int action, int) {
    auto* self = static_cast<MujocoWindow*>(glfwGetWindowUserPointer(window));
    if (!self) return;
    self->handle_mouse_button(button, action);
  }

  static void scroll_callback(GLFWwindow* window, double, double yoffset) {
    auto* self = static_cast<MujocoWindow*>(glfwGetWindowUserPointer(window));
    if (!self) return;
    self->zoom(-0.1 * yoffset);
  }

  void handle_key(int key) {
    switch (key) {
      case GLFW_KEY_ENTER:
      case GLFW_KEY_KP_ENTER:
      case GLFW_KEY_1:
        if (remote_) remote_->pulse_start();
        break;
      case GLFW_KEY_2:
        if (remote_) remote_->pulse_a();
        break;
      case GLFW_KEY_3:
        if (remote_) remote_->pulse_b();
        break;
      case GLFW_KEY_4:
        if (remote_) remote_->pulse_x();
        break;
      case GLFW_KEY_5:
        if (remote_) remote_->pulse_y();
        break;
      case GLFW_KEY_W:
        if (remote_) remote_->adjust_ly(0.1f);
        break;
      case GLFW_KEY_S:
        if (remote_) remote_->adjust_ly(-0.1f);
        break;
      case GLFW_KEY_A:
        if (remote_) remote_->adjust_lx(0.1f);
        break;
      case GLFW_KEY_D:
        if (remote_) remote_->adjust_lx(-0.1f);
        break;
      case GLFW_KEY_Q:
        if (remote_) remote_->adjust_rx(0.1f);
        break;
      case GLFW_KEY_E:
        if (remote_) remote_->adjust_rx(-0.1f);
        break;
      case GLFW_KEY_SPACE:
      case GLFW_KEY_0:
        if (remote_) remote_->zero_axes();
        break;
      case GLFW_KEY_7:
        if (elastic_length_) {
          *elastic_length_ -= 0.1;
          std::cout << "[SDK2 Bridge/CPP] elastic length=" << *elastic_length_ << " lift/more support\n";
        }
        break;
      case GLFW_KEY_8:
        if (elastic_length_) {
          *elastic_length_ += 0.1;
          std::cout << "[SDK2 Bridge/CPP] elastic length=" << *elastic_length_ << " lower/less support\n";
        }
        break;
      case GLFW_KEY_9:
        if (elastic_enabled_) {
          *elastic_enabled_ = !*elastic_enabled_;
          std::cout << "[SDK2 Bridge/CPP] elastic band " << (*elastic_enabled_ ? "on" : "off") << "\n";
        }
        break;
      case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
        break;
      default:
        break;
    }
  }

  void handle_mouse_button(int button, int action) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) left_down_ = action == GLFW_PRESS;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) right_down_ = action == GLFW_PRESS;
    glfwGetCursorPos(window_, &last_x_, &last_y_);
  }

  void handle_cursor(double xpos, double ypos) {
    double dx = xpos - last_x_;
    double dy = ypos - last_y_;
    last_x_ = xpos;
    last_y_ = ypos;
    if (left_down_) {
      cam_.azimuth -= 0.25 * dx;
      cam_.elevation = std::clamp(cam_.elevation - 0.25 * dy, -89.0, 89.0);
    } else if (right_down_) {
      zoom(0.01 * dy);
    }
  }

  void zoom(double amount) {
    cam_.distance = std::clamp(cam_.distance * (1.0 + amount), 0.5, 10.0);
  }

  mjModel* model_;
  mjData* data_;
  bool disabled_;
  TerminalRemote* remote_ = nullptr;
  bool* elastic_enabled_ = nullptr;
  double* elastic_length_ = nullptr;
  GLFWwindow* window_ = nullptr;
  bool left_down_ = false;
  bool right_down_ = false;
  double last_x_ = 0.0;
  double last_y_ = 0.0;
  mjvCamera cam_{};
  mjvOption opt_{};
  mjvScene scn_{};
  mjrContext con_{};
};
#else
class MujocoWindow {
 public:
  MujocoWindow(mjModel*, mjData*, bool disabled, TerminalRemote*, bool*, double*) {
    if (!disabled) std::cout << "[viewer] GLFW support is not compiled; running headless.\n";
  }
  bool running() const { return true; }
  void poll() {}
  void render() {}
};
#endif

class Sdk2MujocoBridge {
 public:
  Sdk2MujocoBridge(std::string config_name,
                   std::string net,
                   int domain_id,
                   std::string input_mode,
                   std::string joystick_type,
                   bool render,
                   bool elastic_band,
                   bool clamp_ctrl,
                   bool debug_lowcmd)
      : g1_dir_(deploy_dir()),
        cfg_(g1::load_config(resolve_config_path(g1_dir_, config_name), g1_dir_)),
        net_(std::move(net)),
        domain_id_(domain_id),
        input_mode_(std::move(input_mode)),
        joystick_type_(std::move(joystick_type)),
        render_(render),
        elastic_band_(elastic_band),
        clamp_ctrl_(clamp_ctrl),
        debug_lowcmd_(debug_lowcmd) {
    load_model();
    init_dds();
    joystick_ = std::make_unique<LinuxJoystick>(input_mode_ == "gamepad", joystick_type_, &remote_);
  }

  ~Sdk2MujocoBridge() {
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
  }

  void run() {
    std::cout << "[SDK2 Bridge/CPP] domain=" << domain_id_ << " net=" << net_
              << " publish " << cfg_.lowstate_topic << "/rt/sportmodestate"
              << " subscribe " << cfg_.lowcmd_topic << "\n";
    std::cout << "[SDK2 Bridge/CPP] render=" << (render_ ? "on" : "off")
              << " ctrlrange clamp=" << (clamp_ctrl_ ? "on" : "off")
              << " elastic_band=" << (elastic_band_ ? "on" : "off") << "\n";
    std::cout << "[SDK2 Bridge/CPP] keys: Enter/1=start, 2=A, 3=B, 4=X, 5=Y; axes W/S A/D Q/E; Space=zero\n";
    std::cout << "[SDK2 Bridge/CPP] viewer elastic keys: 7=lift/more support, 8=lower/less support, 9=toggle, Esc=close\n";
    MujocoWindow window(model_, data_, !render_, &remote_, &elastic_band_, &elastic_length_);
    double last_print = wall_time();
    int render_decimation = std::max(1, static_cast<int>(std::round(0.02 / cfg_.sim_dt)));
    int step_count = 0;
    while (!remote_.exit_requested() && window.running()) {
      double start = wall_time();
      window.poll();
      remote_.poll();
      if (joystick_) joystick_->poll();
      apply_elastic_band();
      apply_lowcmd_control();
      mj_step(model_, data_);
      publish_lowstate();
      publish_sportmode();
      if (step_count % render_decimation == 0) window.render();
      if (wall_time() - last_print > 1.0) {
        double age = last_cmd_time_ > 0.0 ? wall_time() - last_cmd_time_ : 999.0;
        const RemotePacket& rp = remote_.packet();
        std::cout << "[SDK2 Bridge/CPP] t=" << data_->time << "s height=" << data_->qpos[2]
                  << " cmd_age=" << age << "s"
                  << " keys=0x" << std::hex << remote_.key_mask() << std::dec
                  << " axes=[" << rp.lx << "," << rp.ly << "," << rp.rx << "]\n";
        last_print = wall_time();
      }
      ++step_count;
      double sleep_s = cfg_.sim_dt - (wall_time() - start);
      if (sleep_s > 0.0) std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
    }
  }

 private:
  void load_model() {
    fs::path xml = resolve_asset_path(g1_dir_, cfg_.xml_path);
    char error[1024] = {};
    model_ = mj_loadXML(xml.string().c_str(), nullptr, error, sizeof(error));
    if (!model_) throw std::runtime_error("Failed to load MuJoCo XML: " + xml.string() + " " + error);
    data_ = mj_makeData(model_);
    model_->opt.timestep = cfg_.sim_dt;
    num_motor_ = model_->nu;
    low_cmd_q_.assign(static_cast<size_t>(num_motor_), 0.0f);
    low_cmd_dq_.assign(static_cast<size_t>(num_motor_), 0.0f);
    low_cmd_kp_.assign(static_cast<size_t>(num_motor_), 0.0f);
    low_cmd_kd_.assign(static_cast<size_t>(num_motor_), 0.0f);
    low_cmd_tau_.assign(static_cast<size_t>(num_motor_), 0.0f);
    default_q_mj_.assign(static_cast<size_t>(num_motor_), 0.0f);
    default_kp_mj_.assign(static_cast<size_t>(num_motor_), 60.0f);
    default_kd_mj_.assign(static_cast<size_t>(num_motor_), 2.0f);
    qpos_addr_.resize(static_cast<size_t>(num_motor_));
    qvel_addr_.resize(static_cast<size_t>(num_motor_));
    for (int i = 0; i < num_motor_; ++i) {
      int jid = mj_name2id(model_, mjOBJ_JOINT, cfg_.joint_names_mujoco.at(static_cast<size_t>(i)).c_str());
      if (jid < 0) throw std::runtime_error("Joint not found: " + cfg_.joint_names_mujoco.at(static_cast<size_t>(i)));
      qpos_addr_[static_cast<size_t>(i)] = model_->jnt_qposadr[jid];
      qvel_addr_[static_cast<size_t>(i)] = model_->jnt_dofadr[jid];
    }
    std::vector<float> default_mj = cfg_.default_joint_pos;
    std::vector<float> kp_mj = cfg_.kps;
    std::vector<float> kd_mj = cfg_.kds;
    if (!cfg_.isaac_to_mujoco_map.empty()) {
      default_mj.resize(cfg_.isaac_to_mujoco_map.size());
      kp_mj.resize(cfg_.isaac_to_mujoco_map.size());
      kd_mj.resize(cfg_.isaac_to_mujoco_map.size());
      for (size_t i = 0; i < cfg_.isaac_to_mujoco_map.size(); ++i) {
        default_mj[i] = cfg_.default_joint_pos.at(static_cast<size_t>(cfg_.isaac_to_mujoco_map[i]));
        if (cfg_.kps.size() > static_cast<size_t>(cfg_.isaac_to_mujoco_map[i])) {
          kp_mj[i] = cfg_.kps.at(static_cast<size_t>(cfg_.isaac_to_mujoco_map[i]));
        }
        if (cfg_.kds.size() > static_cast<size_t>(cfg_.isaac_to_mujoco_map[i])) {
          kd_mj[i] = cfg_.kds.at(static_cast<size_t>(cfg_.isaac_to_mujoco_map[i]));
        }
      }
    }
    for (int i = 0; i < num_motor_; ++i) {
      size_t idx = static_cast<size_t>(i);
      default_q_mj_[idx] = default_mj.at(idx);
      if (kp_mj.size() > idx) default_kp_mj_[idx] = kp_mj[idx];
      if (kd_mj.size() > idx) default_kd_mj_[idx] = kd_mj[idx];
      data_->qpos[qpos_addr_[idx]] = default_q_mj_[idx];
    }
    data_->qpos[2] = cfg_.init_height;
    anchor_body_id_ = mj_name2id(model_, mjOBJ_BODY, cfg_.anchor_body_name.empty() ? "torso_link" : cfg_.anchor_body_name.c_str());
    if (anchor_body_id_ < 0) anchor_body_id_ = 1;
    elastic_body_id_ = mj_name2id(model_, mjOBJ_BODY, "torso_link");
    if (elastic_body_id_ < 0) elastic_body_id_ = anchor_body_id_;
    mj_forward(model_, data_);
  }

  void init_dds() {
    ChannelFactory::Instance()->Init(domain_id_, net_);
    lowstate_pub_ = std::make_unique<ChannelPublisher<LowStateHg>>(cfg_.lowstate_topic);
    lowstate_pub_->InitChannel();
    sport_pub_ = std::make_unique<ChannelPublisher<SportModeState>>("rt/sportmodestate");
    sport_pub_->InitChannel();
    lowcmd_sub_ = std::make_unique<ChannelSubscriber<LowCmdHg>>(cfg_.lowcmd_topic);
    lowcmd_sub_->InitChannel([this](const void* msg) { lowcmd_handler(msg); }, 10);
  }

  void lowcmd_handler(const void* message) {
    const auto& cmd = *static_cast<const LowCmdHg*>(message);
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    for (int i = 0; i < num_motor_; ++i) {
      const auto& motor = cmd.motor_cmd().at(static_cast<size_t>(i));
      low_cmd_q_[static_cast<size_t>(i)] = motor.q();
      low_cmd_dq_[static_cast<size_t>(i)] = motor.dq();
      low_cmd_kp_[static_cast<size_t>(i)] = motor.kp();
      low_cmd_kd_[static_cast<size_t>(i)] = motor.kd();
      low_cmd_tau_[static_cast<size_t>(i)] = motor.tau();
    }
    low_cmd_received_ = true;
    low_state_.mode_pr() = cmd.mode_pr();
    low_state_.mode_machine() = cmd.mode_machine();
    last_cmd_time_ = wall_time();
  }

  void apply_lowcmd_control() {
    std::vector<float> q_des, dq_des, kp, kd, tau;
    {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      if (!low_cmd_received_) {
        apply_default_hold_control();
        return;
      }
      q_des = low_cmd_q_;
      dq_des = low_cmd_dq_;
      kp = low_cmd_kp_;
      kd = low_cmd_kd_;
      tau = low_cmd_tau_;
    }
    int clipped = 0;
    float max_raw = 0.0f;
    int max_i = 0;
    for (int i = 0; i < num_motor_; ++i) {
      float q = static_cast<float>(data_->qpos[qpos_addr_[static_cast<size_t>(i)]]);
      float dq = static_cast<float>(data_->qvel[qvel_addr_[static_cast<size_t>(i)]]);
      float raw = tau[static_cast<size_t>(i)] +
                  kp[static_cast<size_t>(i)] * (q_des[static_cast<size_t>(i)] - q) +
                  kd[static_cast<size_t>(i)] * (dq_des[static_cast<size_t>(i)] - dq);
      float ctrl = raw;
      if (clamp_ctrl_) {
        double lo = model_->actuator_ctrlrange[2 * i];
        double hi = model_->actuator_ctrlrange[2 * i + 1];
        ctrl = static_cast<float>(std::clamp(static_cast<double>(raw), lo, hi));
        if (std::abs(ctrl - raw) > 1.0e-5f) ++clipped;
      }
      data_->ctrl[i] = ctrl;
      if (std::abs(raw) > std::abs(max_raw)) {
        max_raw = raw;
        max_i = i;
      }
    }
    if (debug_lowcmd_ && wall_time() - last_debug_time_ > 0.5) {
      const char* name = mj_id2name(model_, mjOBJ_ACTUATOR, max_i);
      std::cout << "[LowCmdDebug/CPP] raw_max=" << (name ? name : "?") << ":" << max_raw
                << " clipped=" << clipped << "/" << num_motor_ << "\n";
      last_debug_time_ = wall_time();
    }
  }

  void apply_default_hold_control() {
    for (int i = 0; i < num_motor_; ++i) {
      size_t idx = static_cast<size_t>(i);
      float q = static_cast<float>(data_->qpos[qpos_addr_[idx]]);
      float dq = static_cast<float>(data_->qvel[qvel_addr_[idx]]);
      float raw = default_kp_mj_[idx] * (default_q_mj_[idx] - q) - default_kd_mj_[idx] * dq;
      if (clamp_ctrl_) {
        double lo = model_->actuator_ctrlrange[2 * i];
        double hi = model_->actuator_ctrlrange[2 * i + 1];
        raw = static_cast<float>(std::clamp(static_cast<double>(raw), lo, hi));
      }
      data_->ctrl[i] = raw;
    }
  }

  void apply_elastic_band() {
    std::fill(data_->xfrc_applied, data_->xfrc_applied + 6 * model_->nbody, 0.0);
    if (!elastic_band_) return;
    const double* pos = data_->xpos + 3 * elastic_body_id_;
    const double* vel = data_->cvel + 6 * elastic_body_id_ + 3;
    std::array<double, 3> delta{0.0 - pos[0], 0.0 - pos[1], 3.0 - pos[2]};
    double distance = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    if (distance < 1.0e-6) return;
    std::array<double, 3> direction{delta[0] / distance, delta[1] / distance, delta[2] / distance};
    double velocity = vel[0] * direction[0] + vel[1] * direction[1] + vel[2] * direction[2];
    double force = elastic_stiffness_ * (distance - elastic_length_) - elastic_damping_ * velocity;
    double* xfrc = data_->xfrc_applied + 6 * elastic_body_id_;
    xfrc[0] = force * direction[0];
    xfrc[1] = force * direction[1];
    xfrc[2] = force * direction[2];
  }

  void publish_lowstate() {
    low_state_.tick() = low_state_.tick() + 1;
    for (int i = 0; i < num_motor_; ++i) {
      auto& motor = low_state_.motor_state().at(static_cast<size_t>(i));
      motor.mode() = 1;
      motor.q() = static_cast<float>(data_->qpos[qpos_addr_[static_cast<size_t>(i)]]);
      motor.dq() = static_cast<float>(data_->qvel[qvel_addr_[static_cast<size_t>(i)]]);
      motor.tau_est() = static_cast<float>(data_->actuator_force[i]);
    }
    auto& imu = low_state_.imu_state();
    imu.quaternion() = {static_cast<float>(data_->qpos[3]), static_cast<float>(data_->qpos[4]),
                        static_cast<float>(data_->qpos[5]), static_cast<float>(data_->qpos[6])};
    imu.gyroscope() = {static_cast<float>(data_->qvel[3]), static_cast<float>(data_->qvel[4]), static_cast<float>(data_->qvel[5])};
    imu.accelerometer() = {0.0f, 0.0f, 0.0f};
    auto rpy = quat_to_rpy(data_->qpos + 3);
    imu.rpy() = rpy;
    RemotePacket packet = remote_.packet();
    std::memcpy(low_state_.wireless_remote().data(), &packet, sizeof(packet));
    low_state_.crc() = crc32_core(reinterpret_cast<uint32_t*>(&low_state_), (sizeof(LowStateHg) >> 2) - 1);
    lowstate_pub_->Write(low_state_);
  }

  void publish_sportmode() {
    const double* pos = data_->xpos + 3 * anchor_body_id_;
    const double* quat = data_->xquat + 4 * anchor_body_id_;
    sport_state_.position() = {static_cast<float>(pos[0]), static_cast<float>(pos[1]), static_cast<float>(pos[2])};
    sport_state_.imu_state().quaternion() = {static_cast<float>(quat[0]), static_cast<float>(quat[1]),
                                             static_cast<float>(quat[2]), static_cast<float>(quat[3])};
    sport_state_.velocity() = {static_cast<float>(data_->qvel[0]), static_cast<float>(data_->qvel[1]), static_cast<float>(data_->qvel[2])};
    sport_pub_->Write(sport_state_);
  }

  fs::path g1_dir_;
  g1::Config cfg_;
  std::string net_;
  int domain_id_ = 1;
  std::string input_mode_ = "keyboard";
  std::string joystick_type_ = "switch";
  bool render_ = true;
  bool elastic_band_ = false;
  bool clamp_ctrl_ = true;
  bool debug_lowcmd_ = false;
  double elastic_stiffness_ = 200.0;
  double elastic_damping_ = 100.0;
  double elastic_length_ = 0.0;
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  int num_motor_ = 0;
  int anchor_body_id_ = 1;
  int elastic_body_id_ = 1;
  std::vector<int> qpos_addr_;
  std::vector<int> qvel_addr_;
  std::mutex cmd_mutex_;
  bool low_cmd_received_ = false;
  std::vector<float> low_cmd_q_;
  std::vector<float> low_cmd_dq_;
  std::vector<float> low_cmd_kp_;
  std::vector<float> low_cmd_kd_;
  std::vector<float> low_cmd_tau_;
  std::vector<float> default_q_mj_;
  std::vector<float> default_kp_mj_;
  std::vector<float> default_kd_mj_;
  double last_cmd_time_ = 0.0;
  double last_debug_time_ = 0.0;
  TerminalRemote remote_;
  std::unique_ptr<LinuxJoystick> joystick_;
  LowStateHg low_state_;
  SportModeState sport_state_;
  std::unique_ptr<ChannelPublisher<LowStateHg>> lowstate_pub_;
  std::unique_ptr<ChannelPublisher<SportModeState>> sport_pub_;
  std::unique_ptr<ChannelSubscriber<LowCmdHg>> lowcmd_sub_;
};

}  // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::string config = "g1_walk.yaml";
  std::string net = "lo";
  std::string input_mode = "keyboard";
  std::string joystick_type = "switch";
  int domain = 1;
  bool render = true;
  bool elastic_band = false;
  bool clamp_ctrl = true;
  bool debug_lowcmd = false;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    auto value = [&](const std::string& flag) {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
      return std::string(argv[++i]);
    };
    if (a == "--config") config = value(a);
    else if (a == "--net") net = value(a);
    else if (a == "--domain_id") domain = std::stoi(value(a));
    else if (a == "--no_render") render = false;
    else if (a == "--no_clamp_ctrl") clamp_ctrl = false;
    else if (a == "--debug_lowcmd") debug_lowcmd = true;
    else if (a == "--input") input_mode = value(a);
    else if (a == "--joystick_type") joystick_type = value(a);
    else if (a == "--elastic_band") elastic_band = true;
    else if (a == "-h" || a == "--help") {
      std::cout << "Usage: " << argv[0]
                << " [--config YAML] [--net IFACE] [--domain_id ID] [--input keyboard|gamepad]"
                << " [--joystick_type switch|xbox] [--no_render] [--elastic_band] [--debug_lowcmd]\n";
      return 0;
    }
  }
  try {
    Sdk2MujocoBridge bridge(config, net, domain, input_mode, joystick_type, render, elastic_band, clamp_ctrl, debug_lowcmd);
    bridge.run();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << "\n";
    return 1;
  }
}
