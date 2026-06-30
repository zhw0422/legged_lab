cmake_minimum_required(VERSION 3.16)
project(g1_cpp_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(G1_CPP_BUILD_SDK2 "Build local unitree_sdk2 target as part of g1_cpp" OFF)
option(G1_CPP_ENABLE_VIEWER "Enable MuJoCo GLFW viewer" ON)

if(G1_CPP_BUILD_SDK2)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(unitree_sdk2 EXCLUDE_FROM_ALL)
endif()

find_package(yaml-cpp REQUIRED)

set(G1_CPP_MUJOCO_HINTS
  "$ENV{MUJOCO_ROOT}"
  "${CMAKE_CURRENT_SOURCE_DIR}/mujoco"
  "$ENV{HOME}/.mujoco/mujoco-3.3.6"
)

find_package(mujoco CONFIG QUIET
  HINTS ${G1_CPP_MUJOCO_HINTS}
  PATH_SUFFIXES lib/cmake/mujoco cmake/mujoco)
if(NOT mujoco_FOUND)
  find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
    HINTS ${G1_CPP_MUJOCO_HINTS}
    PATH_SUFFIXES include)
  find_library(MUJOCO_LIBRARY mujoco
    HINTS ${G1_CPP_MUJOCO_HINTS}
    PATH_SUFFIXES lib lib64 bin)
  if(NOT MUJOCO_INCLUDE_DIR OR NOT MUJOCO_LIBRARY)
    message(FATAL_ERROR
      "MuJoCo not found. Expected one of: $MUJOCO_ROOT, "
      "${CMAKE_CURRENT_SOURCE_DIR}/mujoco, or ~/.mujoco/mujoco-3.3.6. "
      "You can also set MUJOCO_INCLUDE_DIR and MUJOCO_LIBRARY manually.")
  endif()
  add_library(mujoco::mujoco UNKNOWN IMPORTED)
  set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}")
endif()

set(G1_CPP_ONNXRUNTIME_HINTS
  "$ENV{ONNXRUNTIME_ROOT}"
  "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime"
  "$ENV{HOME}/.onnx/onnxruntime-linux-x64-1.22.0"
)

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${G1_CPP_ONNXRUNTIME_HINTS}
  PATH_SUFFIXES include)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS ${G1_CPP_ONNXRUNTIME_HINTS}
  PATH_SUFFIXES lib lib64)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime C++ API not found. Expected one of: $ONNXRUNTIME_ROOT, "
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime, or ~/.onnx/onnxruntime-linux-x64-1.22.0. "
    "Check that include/onnxruntime_cxx_api.h and lib/libonnxruntime.so exist.")
endif()
add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

if(G1_CPP_ENABLE_VIEWER)
  find_package(glfw3 QUIET)
  if(glfw3_FOUND)
    set(G1_CPP_VIEWER_LIB glfw)
    add_compile_definitions(G1_CPP_HAS_GLFW=1)
  else()
    message(WARNING "glfw3 not found; C++ sim2sim will build in headless mode only.")
  endif()
endif()

add_library(g1_cpp_sim2sim
  src/config.cpp
  src/controllers.cpp
  src/math_utils.cpp
  src/onnx_policy.cpp
  src/sim2sim_controller.cpp
)
target_include_directories(g1_cpp_sim2sim PUBLIC include)
target_compile_definitions(g1_cpp_sim2sim PUBLIC
  G1_CPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(g1_cpp_sim2sim PUBLIC
  yaml-cpp
  mujoco::mujoco
  onnxruntime::onnxruntime
  ${G1_CPP_VIEWER_LIB}
)
if(G1_CPP_BUILD_SDK2)
  target_link_libraries(g1_cpp_sim2sim PUBLIC unitree_sdk2)
endif()

foreach(name walk amp mimic attention sdk2_bridge)
  add_executable(sim2sim_${name} src/sim2sim_${name}.cpp)
  target_link_libraries(sim2sim_${name} PRIVATE g1_cpp_sim2sim)
endforeach()
