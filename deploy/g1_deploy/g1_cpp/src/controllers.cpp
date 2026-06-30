#include "g1_cpp/controllers.hpp"

#include <sys/select.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace g1 {

namespace {

double now_seconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

float clamp(float value, Range r) {
  return std::max(r.lo, std::min(r.hi, value));
}

}  // namespace

KeyboardController::KeyboardController(Range vx, Range vy, Range vyaw)
    : vx_range_(vx), vy_range_(vy), vyaw_range_(vyaw) {}

KeyboardController::~KeyboardController() {
  stop();
}

void KeyboardController::start() {
  if (!isatty(STDIN_FILENO)) {
    throw std::runtime_error("Keyboard input requires an interactive terminal. Use --input const for headless runs.");
  }
  if (tcgetattr(STDIN_FILENO, &old_termios_) == 0) {
    termios raw = old_termios_;
    raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    raw_enabled_ = true;
  }
  running_ = true;
  thread_ = std::thread(&KeyboardController::loop, this);
}

void KeyboardController::stop() {
  running_ = false;
  if (thread_.joinable()) thread_.join();
  if (raw_enabled_) {
    tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);
    raw_enabled_ = false;
  }
}

VelocityCommand KeyboardController::velocity() {
  std::lock_guard<std::mutex> lock(mutex_);
  return cmd_;
}

void KeyboardController::loop() {
  while (running_ && !exit_requested_) {
    fd_set set;
    FD_ZERO(&set);
    FD_SET(STDIN_FILENO, &set);
    timeval tv{0, 20000};
    if (select(STDIN_FILENO + 1, &set, nullptr, nullptr, &tv) > 0) {
      char c = 0;
      if (read(STDIN_FILENO, &c, 1) == 1) apply_key(c);
    }
  }
}

void KeyboardController::apply_key(char key) {
  std::lock_guard<std::mutex> lock(mutex_);
  switch (key) {
    case 'w':
    case 'W':
      cmd_.vx += 0.1f;
      break;
    case 's':
    case 'S':
      cmd_.vx -= 0.1f;
      break;
    case 'a':
    case 'A':
      cmd_.vy += 0.05f;
      break;
    case 'd':
    case 'D':
      cmd_.vy -= 0.05f;
      break;
    case 'q':
    case 'Q':
      cmd_.vyaw += 0.1f;
      break;
    case 'e':
    case 'E':
      cmd_.vyaw -= 0.1f;
      break;
    case ' ':
    case '0':
      cmd_ = {};
      break;
    case '1':
    case '2':
    case '3':
    case '4':
      active_policy_ = key - '0';
      break;
    case 'x':
    case 'X':
    case 27:
      exit_requested_ = true;
      running_ = false;
      break;
    default:
      break;
  }
  cmd_.vx = clamp(cmd_.vx, vx_range_);
  cmd_.vy = clamp(cmd_.vy, vy_range_);
  cmd_.vyaw = clamp(cmd_.vyaw, vyaw_range_);
}

ConstantController::ConstantController(float vx, float vy, float vyaw, float warmup_s, float ramp_s)
    : target_{vx, vy, vyaw}, warmup_s_(warmup_s), ramp_s_(std::max(ramp_s, 1e-3f)) {}

void ConstantController::start() {
  start_time_ = now_seconds();
}

VelocityCommand ConstantController::velocity() {
  double elapsed = now_seconds() - start_time_;
  if (elapsed < warmup_s_) return {};
  float alpha = std::min<float>((elapsed - warmup_s_) / ramp_s_, 1.0f);
  return {target_.vx * alpha, target_.vy * alpha, target_.vyaw * alpha};
}

}  // namespace g1
