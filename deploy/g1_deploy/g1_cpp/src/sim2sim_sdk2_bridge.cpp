#include "g1_cpp/config.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  std::string config = "g1_walk.yaml";
  std::string net = "lo";
  int domain = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    auto value = [&](const std::string& flag) {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
      return std::string(argv[++i]);
    };
    if (a == "--config") config = value(a);
    else if (a == "--net") net = value(a);
    else if (a == "--domain_id") domain = std::stoi(value(a));
    else if (a == "-h" || a == "--help") {
      std::cout << "Usage: " << argv[0] << " [--config YAML] [--net IFACE] [--domain_id ID]\n";
      return 0;
    }
  }

  std::cout << "C++ SDK2 bridge scaffold\n"
            << "  config: " << config << "\n"
            << "  net: " << net << " domain_id: " << domain << "\n"
            << "The Python bridge maps LowCmd/LowState into MuJoCo. The C++ tree now vendors\n"
            << "unitree_sdk2 under g1_cpp/unitree_sdk2 and builds the target with\n"
            << "-DG1_CPP_BUILD_SDK2=ON; DDS LowCmd/LowState parity can be completed here.\n";
  return 0;
}
