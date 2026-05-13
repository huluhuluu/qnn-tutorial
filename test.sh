#!/usr/bin/env bash

set -euo pipefail

SLEEP_SECONDS=3
ADB_SERIAL="127.0.0.1:40404"
TEST_KIND="${1:-static}"
BUILD_DIR="build"
ANDROID_ABI="arm64-v8a"
ANDROID_PLATFORM="android-28"

print_usage() {
  cat <<'EOF'
Usage: ./test.sh [static|switch|all|--help]

  static   Run qnn_weight_static_benchmark only. Default.
  switch   Run qnn_weight_switch_benchmark only.
  all      Run both benchmarks for the same shape set.
  --help   Show this help message.
EOF
}

require_env() {
  if [ -z "${ANDROID_NDK_ROOT:-}" ]; then
    printf 'ANDROID_NDK_ROOT is not set\n' >&2
    exit 1
  fi
  if [ -z "${QNN_SDK_ROOT:-}" ]; then
    printf 'QNN_SDK_ROOT is not set\n' >&2
    exit 1
  fi
}

configure_build() {
  cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DANDROID_PLATFORM="${ANDROID_PLATFORM}"
}

device_root_for() {
  local benchmark_bin="$1"
  case "${benchmark_bin}" in
    qnn_weight_static_benchmark)
      printf '/data/local/tmp/qnn_matmul_weight_static'
      ;;
    qnn_weight_switch_benchmark)
      printf '/data/local/tmp/qnn_static_binary_switch'
      ;;
    *)
      printf 'Unknown benchmark binary: %s\n' "${benchmark_bin}" >&2
      exit 1
      ;;
  esac
}

host_log_for() {
  local benchmark_bin="$1"
  case "${benchmark_bin}" in
    qnn_weight_static_benchmark)
      printf 'test-qnn-static.log'
      ;;
    qnn_weight_switch_benchmark)
      printf 'test-qnn-switch.log'
      ;;
    *)
      printf 'Unknown benchmark binary: %s\n' "${benchmark_bin}" >&2
      exit 1
      ;;
  esac
}

build_benchmark() {
  local benchmark_bin="$1"
  configure_build
  cmake --build "${BUILD_DIR}" --target "${benchmark_bin}" -j4
}

push_benchmark() {
  local benchmark_bin="$1"
  local device_root
  device_root="$(device_root_for "${benchmark_bin}")"

  adb -s "${ADB_SERIAL}" shell "mkdir -p ${device_root}"
  adb -s "${ADB_SERIAL}" push \
    "${BUILD_DIR}/${benchmark_bin}" \
    "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
    "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so" \
    "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so" \
    "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV79Stub.so" \
    "${QNN_SDK_ROOT}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" \
    "${device_root}/"
}

run_case() {
  local benchmark_bin="$1"
  shift
  local prompt_len="$1"
  local k_dim="$2"
  local n_dim="$3"
  local rc=0
  local device_root
  device_root="$(device_root_for "${benchmark_bin}")"

  adb -s "${ADB_SERIAL}" shell "
    cd ${device_root} && \
    export LD_LIBRARY_PATH=\$PWD && \
    export ADSP_LIBRARY_PATH=\$PWD && \
    ./${benchmark_bin} \
      --backend ./libQnnHtp.so \
      --m ${prompt_len} --k ${k_dim} --n ${n_dim} \
      --warmup 3 --iters 5 \
      --precisions fp32,fp16,int8,int16
  " || rc=$?
  if [ "${rc}" -ne 0 ]; then
    printf 'CASE FAILED: bin=%s prompt_len=%s k=%s n=%s exit_code=%s\n' \
      "${benchmark_bin}" "${prompt_len}" "${k_dim}" "${n_dim}" "${rc}"
  fi
  sleep "${SLEEP_SECONDS}"
  return 0
}

run_group() {
  local benchmark_bin="$1"
  shift
  local prompt_len="$1"

  printf '\n####### bench = %s prompt len = %s #######\n' "${benchmark_bin}" "${prompt_len}"
  run_case "${benchmark_bin}" "${prompt_len}" 2048 2048
  run_case "${benchmark_bin}" "${prompt_len}" 2048 6144
  run_case "${benchmark_bin}" "${prompt_len}" 6144 2048
  # if [ "$prompt_len" -le 128 ]; then
  #   run_case "${benchmark_bin}" "${prompt_len}" 151936 2048
  # fi
}

run_benchmark() {
  local benchmark_bin="$1"
  printf '\n===== build + push: %s =====\n' "${benchmark_bin}"
  build_benchmark "${benchmark_bin}"
  push_benchmark "${benchmark_bin}"
  for prompt_len in 1 2 4 8 16 32 64 128 256 512 1024 2048; do
    run_group "${benchmark_bin}" "${prompt_len}"
  done
}

run_benchmark_logged() {
  local benchmark_bin="$1"
  local host_log
  host_log="$(host_log_for "${benchmark_bin}")"
  printf 'host log: %s\n' "${host_log}"
  run_benchmark "${benchmark_bin}" 2>&1 | tee "${host_log}"
}

case "${TEST_KIND}" in
  static)
    require_env
    run_benchmark_logged "qnn_weight_static_benchmark"
    ;;
  switch)
    require_env
    run_benchmark_logged "qnn_weight_switch_benchmark"
    ;;
  all)
    require_env
    run_benchmark_logged "qnn_weight_static_benchmark"
    run_benchmark_logged "qnn_weight_switch_benchmark"
    ;;
  --help|-h|help)
    print_usage
    ;;
  *)
    printf 'Unknown test kind: %s\n\n' "${TEST_KIND}" >&2
    print_usage >&2
    exit 1
    ;;
esac
