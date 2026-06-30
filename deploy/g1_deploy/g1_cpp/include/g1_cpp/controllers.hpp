#pragma once

#include "g1_cpp/config.hpp"

#include <atomic>
#include <mutex>
#include <termios.h>
#include <thread>

namespace g1 {

struct VelocityCommand {
  float vx = 0.0f;
  float vy = 0.0f;
  float vyaw = 0.0f;
};

class CommandController {
 public:
  virtual ~CommandController() = default;
  virtual void start() {}
  virtual void stop() {}
  virtual VelocityCommand velocity() = 0;
  virtual int active_policy() const { return active_policy_; }
  virtual bool exit_requested() const { return exit_requested_; }

 protected:
  int active_policy_ = 1;
  bool exit_requested_ = false;
};

class KeyboardController final : public CommandController {
 public:
  KeyboardController(Range vx, Range vy, Range vyaw);
  ~KeyboardController() override;
  void start() override;
  void stop() override;
  VelocityCommand velocity() override;

 private:
  void loop();
  void apply_key(char key);

  Range vx_range_;
  Range vy_range_;
  Range vyaw_range_;
  VelocityCommand cmd_;
  std::mutex mutex_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  bool raw_enabled_ = false;
  termios old_termios_{};
};

class ConstantController final : public CommandController {
 public:
  ConstantController(float vx, float vy, float vyaw, float warmup_s, float ramp_s);
  void start() override;
  VelocityCommand velocity() override;

 private:
  VelocityCommand target_;
  float warmup_s_;
  float ramp_s_;
  double start_time_ = 0.0;
};

}  // namespace g1
