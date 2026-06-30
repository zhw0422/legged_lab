#include "g1_cpp/onnx_policy.hpp"

#include <array>
#include <stdexcept>

namespace g1 {

Ort::Env& OnnxPolicy::env() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "g1_cpp_sim2sim");
  return env;
}

OnnxPolicy::OnnxPolicy(const std::string& path) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_ = Ort::Session(env(), path.c_str(), options);

  size_t n_inputs = session_.GetInputCount();
  size_t n_outputs = session_.GetOutputCount();
  for (size_t i = 0; i < n_inputs; ++i) {
    auto name = session_.GetInputNameAllocated(i, allocator_);
    input_names_.push_back(name.get());
  }
  for (size_t i = 0; i < n_outputs; ++i) {
    auto name = session_.GetOutputNameAllocated(i, allocator_);
    output_names_.push_back(name.get());
  }
  for (auto& n : input_names_) input_name_ptrs_.push_back(n.c_str());
  for (auto& n : output_names_) output_name_ptrs_.push_back(n.c_str());

  try {
    auto input_info = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto input_shape = input_info.GetShape();
    if (!input_shape.empty() && input_shape.back() > 0) input_dim_ = static_cast<int>(input_shape.back());
    auto output_info = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    if (!output_shape.empty() && output_shape.back() > 0) output_dim_ = static_cast<int>(output_shape.back());
  } catch (const std::exception&) {
    input_dim_ = -1;
    output_dim_ = -1;
  }
}

std::vector<float> OnnxPolicy::run_single(const std::vector<float>& obs) {
  std::array<int64_t, 2> shape{1, static_cast<int64_t>(obs.size())};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<float>(
      memory, const_cast<float*>(obs.data()), obs.size(), shape.data(), shape.size());
  auto outputs = session_.Run(Ort::RunOptions{nullptr},
                              input_name_ptrs_.data(), &input, 1,
                              output_name_ptrs_.data(), output_name_ptrs_.size());
  return tensor_to_vector(outputs.front());
}

std::vector<Ort::Value> OnnxPolicy::run_named(const std::vector<float>& obs, float time_step) {
  if (input_names_.size() < 2) {
    throw std::runtime_error("Tracking ONNX model must expose obs and time_step inputs.");
  }
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<int64_t, 2> obs_shape{1, static_cast<int64_t>(obs.size())};
  std::array<int64_t, 2> ts_shape{1, 1};
  std::vector<float> ts{time_step};
  std::vector<Ort::Value> inputs;
  for (const auto& name : input_names_) {
    if (name == "time_step") {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          memory, ts.data(), ts.size(), ts_shape.data(), ts_shape.size()));
    } else {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          memory, const_cast<float*>(obs.data()), obs.size(), obs_shape.data(), obs_shape.size()));
    }
  }
  return session_.Run(Ort::RunOptions{nullptr},
                      input_name_ptrs_.data(), inputs.data(), inputs.size(),
                      output_name_ptrs_.data(), output_name_ptrs_.size());
}

std::vector<float> tensor_to_vector(const Ort::Value& value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  size_t n = info.GetElementCount();
  if (n > 10000000ULL) {
    throw std::runtime_error("ONNX output tensor element count is unexpectedly large.");
  }
  const float* data = value.GetTensorData<float>();
  return std::vector<float>(data, data + n);
}

}  // namespace g1
