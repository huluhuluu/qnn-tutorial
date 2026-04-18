cmake_minimum_required(VERSION 3.16)
project(qnn_online_matmul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(DEFINED ENV{QNN_SDK_ROOT})
  set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}")
else()
  set(QNN_SDK_ROOT "/root/qairt/2.40.0.251030")
endif()

if(NOT EXISTS "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h")
  message(FATAL_ERROR "QNN_SDK_ROOT is invalid: ${QNN_SDK_ROOT}")
endif()

add_executable(qnn_online_matmul online_matmul.cpp)
target_include_directories(
  qnn_online_matmul
  PRIVATE
    "${QNN_SDK_ROOT}/include"
    "${QNN_SDK_ROOT}/include/QNN")

add_executable(qnn_precision_benchmark precision_benchmark.cpp)
target_include_directories(
  qnn_precision_benchmark
  PRIVATE
    "${QNN_SDK_ROOT}/include"
    "${QNN_SDK_ROOT}/include/QNN")

if(ANDROID)
  target_link_libraries(qnn_online_matmul PRIVATE dl log)
  target_link_libraries(qnn_precision_benchmark PRIVATE dl log)
else()
  target_link_libraries(qnn_online_matmul PRIVATE dl)
  target_link_libraries(qnn_precision_benchmark PRIVATE dl)
endif()
