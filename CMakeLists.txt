cmake_minimum_required(VERSION 3.16)
project(qnn_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(DEFINED ENV{QNN_SDK_ROOT})
  set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}")
else()
  set(QNN_SDK_ROOT "/root/qairt/2.40.0.251030")
endif()

if(NOT EXISTS "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h")
  message(FATAL_ERROR "QNN_SDK_ROOT is invalid: ${QNN_SDK_ROOT}")
endif()

function(add_qnn_example target source)
  add_executable(${target} ${source})
  target_include_directories(
    ${target}
    PRIVATE
      "${QNN_SDK_ROOT}/include"
      "${QNN_SDK_ROOT}/include/QNN")
  if(ANDROID)
    target_link_libraries(${target} PRIVATE dl log)
  else()
    target_link_libraries(${target} PRIVATE dl)
  endif()
endfunction()

add_qnn_example(qnn_online_matmul blog/qnn-setup/online_matmul.cpp)
add_qnn_example(qnn_precision_benchmark blog/qnn-setup/precision_benchmark.cpp)
add_qnn_example(qnn_weight_static_benchmark
                blog/qnn-matmul-weight-static/weight_static_benchmark.cpp)
add_qnn_example(qnn_weight_switch_benchmark
                blog/qnn-matmul-weight-static/weight_switch_benchmark.cpp)
