#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace g1 {

class OnnxPolicy {
 public:
  explicit OnnxPolicy(const std::string& path);

  std::vector<float> run_single(const std::vector<float>& obs);
  std::vector<Ort::Value> run_named(const std::vector<float>& obs, float time_step);

  int input_dim() const { return input_dim_; }
  int output_dim() const { return output_dim_; }
  const std::vector<std::string>& input_names() const { return input_names_; }
  const std::vector<std::string>& output_names() const { return output_names_; }

 private:
  static Ort::Env& env();
  Ort::Session session_{nullptr};
  Ort::AllocatorWithDefaultOptions allocator_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_name_ptrs_;
  std::vector<const char*> output_name_ptrs_;
  int input_dim_ = -1;
  int output_dim_ = -1;
};

std::vector<float> tensor_to_vector(const Ort::Value& value);

}  // namespace g1
