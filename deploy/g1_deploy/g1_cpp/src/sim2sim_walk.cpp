#include "g1_cpp/sim2sim_controller.hpp"

int main(int argc, char** argv) {
  return g1::run_sim2sim_main(argc, argv, g1::PolicyKind::Walk, "g1_walk.yaml", "g1_flat_1.onnx");
}
