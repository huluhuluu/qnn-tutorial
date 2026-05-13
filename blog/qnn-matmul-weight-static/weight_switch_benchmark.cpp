#include <QNN/HTP/QnnHtpDevice.h>
#include <QNN/QnnInterface.h>
#include <QNN/QnnLog.h>
#include <QNN/QnnOpDef.h>
#include <QNN/QnnTypes.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using QnnInterfaceGetProvidersFn_t =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t*** providerList, uint32_t* numProviders);

// This benchmark compares three ways to switch MatMul weights at runtime:
//   1. app_write: one graph, weight is an APP_WRITE input changed per execute().
//   2. static_load: one static-weight graph per weight, serialized to context binary,
//      written to disk, then each iteration re-reads the binary and creates a temporary context.
//   3. update_static: one graph with UPDATEABLE_STATIC weight, then each
//      iteration reads a new weight file and applies it through QnnTensor_updateGraphTensors().
enum class BackendKind { Cpu, Gpu, Htp, Unknown };

enum class PrecisionKind { Fp32, Fp16, Int8, Int16 };

constexpr const char* kModeAppWrite     = "app_write";
constexpr const char* kModeStaticLoad   = "static_load";
constexpr const char* kModeUpdateStatic = "update_static";

struct Shape {
  uint32_t rows;
  uint32_t inner;
  uint32_t cols;
};

struct AppOptions {
  std::string backendPath;
  std::optional<std::string> logPath;
  std::vector<PrecisionKind> precisions = {
      PrecisionKind::Fp16, PrecisionKind::Int8, PrecisionKind::Int16};
  std::vector<Shape> shapes = {{1, 2048, 6144}, {1, 6144, 2048}, {1, 2048, 2048}};
  uint32_t numWeights = 2;
  uint32_t warmup     = 3;
  uint32_t iterations = 5;
};

struct TensorBuffer {
  Qnn_DataType_t dataType             = QNN_DATATYPE_UNDEFINED;
  Qnn_QuantizeParams_t quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
  std::vector<Qnn_ScaleOffset_t> axisScaleOffsets;
  std::vector<uint8_t> storage;
  size_t elements = 0;

  // QNN axis quantization stores a raw pointer into axisScaleOffsets. Refresh it after moves/copies
  // before passing this buffer's quantizeParams to QNN.
  void refreshQuantizationPointers() {
    if (quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      quantizeParams.axisScaleOffsetEncoding.numScaleOffsets =
          static_cast<uint32_t>(axisScaleOffsets.size());
      quantizeParams.axisScaleOffsetEncoding.scaleOffset =
          axisScaleOffsets.empty() ? nullptr : axisScaleOffsets.data();
    }
  }

  void* data() { return storage.empty() ? nullptr : storage.data(); }
  const void* data() const { return storage.empty() ? nullptr : storage.data(); }
  uint32_t bytes() const { return static_cast<uint32_t>(storage.size()); }
};

struct CaseResult {
  Shape shape{};
  PrecisionKind precision = PrecisionKind::Fp16;
  std::string mode;
  std::string status = "OK";
  std::string detail;
  bool validationPassed = false;
  double prepareMs       = 0.0;
  double ssdDataLoadMs   = -1.0;
  double ssdGraphLoadMs  = -1.0;
  double htpWriteMs      = -1.0;
  double finalizeMs      = 0.0;
  double firstMs         = 0.0;
  double avgMs           = 0.0;
  double stdMs           = 0.0;
  double releaseMs       = 0.0;
  double maxAbsError     = 0.0;
  double meanAbsError    = 0.0;
  uint64_t binaryBytes   = 0;
  uint64_t staticBytes   = 0;
  uint64_t runtimeBytes  = 0;
};

struct StaticBinary {
  std::string graphName;
  std::array<uint32_t, 2> inputDims{};
  std::array<uint32_t, 2> outputDims{};
  Qnn_Tensor_t inputTensor  = QNN_TENSOR_INIT;
  Qnn_Tensor_t outputTensor = QNN_TENSOR_INIT;
  std::string binaryPath;
  uint64_t binaryBytes = 0;
  uint64_t staticBytes = 0;
  double finalizeMs    = 0.0;
};

struct LoadedGraph {
  Qnn_ContextHandle_t context = nullptr;
  Qnn_GraphHandle_t graph     = nullptr;
  Qnn_Tensor_t inputTensor    = QNN_TENSOR_INIT;
  Qnn_Tensor_t outputTensor   = QNN_TENSOR_INIT;
};

struct TensorFileArtifacts {
  std::string inputPath;
  std::vector<std::string> weightPaths;
};

struct CleanupRegistry {
  std::vector<std::string> paths;

  void add(const std::string& path) { paths.push_back(path); }

  void addAll(const std::vector<std::string>& extraPaths) {
    paths.insert(paths.end(), extraPaths.begin(), extraPaths.end());
  }

  void cleanup() const {
    for (const auto& path : paths) {
      if (path.empty()) continue;
      std::remove(path.c_str());
    }
  }
};

std::string defaultBackendPath() {
  if (const char* sdkRoot = std::getenv("QNN_SDK_ROOT")) {
    return std::string(sdkRoot) + "/lib/x86_64-linux-clang/libQnnCpu.so";
  }
  return "/root/qairt/2.40.0.251030/lib/x86_64-linux-clang/libQnnCpu.so";
}

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

uint32_t parseUintOption(const char* raw, const std::string& flagName) {
  if (raw == nullptr || *raw == '\0') throw std::runtime_error(flagName + " requires a value");
  char* end = nullptr;
  auto value = std::strtoul(raw, &end, 10);
  if (end == raw || *end != '\0') throw std::runtime_error("invalid value for " + flagName);
  return static_cast<uint32_t>(value);
}

const char* precisionKindName(PrecisionKind precision) {
  switch (precision) {
    case PrecisionKind::Fp32: return "fp32";
    case PrecisionKind::Fp16: return "fp16";
    case PrecisionKind::Int8: return "int8";
    case PrecisionKind::Int16: return "int16";
  }
  return "unknown";
}

PrecisionKind parsePrecisionKind(const std::string& raw) {
  const std::string value = toLower(raw);
  if (value == "fp32") return PrecisionKind::Fp32;
  if (value == "fp16") return PrecisionKind::Fp16;
  if (value == "int8") return PrecisionKind::Int8;
  if (value == "int16") return PrecisionKind::Int16;
  throw std::runtime_error("unknown precision: " + raw);
}

std::vector<PrecisionKind> parsePrecisionList(const std::string& raw) {
  std::vector<PrecisionKind> precisions;
  size_t begin = 0;
  while (begin < raw.size()) {
    const size_t end = raw.find(',', begin);
    const std::string token =
        raw.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (!token.empty()) precisions.push_back(parsePrecisionKind(token));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  if (precisions.empty()) throw std::runtime_error("--precisions produced an empty list");
  return precisions;
}

void printUsage(const char* programName) {
  std::cout
      << "Usage: " << programName
      << " [--backend <path>] [--m <rows> --k <inner> --n <cols>]\n"
      << "       [--num-weights <count>] [--warmup <count>] [--iters <count>]\n"
      << "       [--precisions fp16,int8,int16] [--log <path>]\n"
      << "       [backend_path]\n"
      << "Default shapes: 1x2048x6144, 1x6144x2048, 1x2048x2048.\n"
      << "Default precisions: fp16,int8,int16.\n"
      << "No log file is written unless --log is provided.\n";
}

AppOptions parseCommandLine(int argc, char** argv) {
  AppOptions options{};
  options.backendPath = defaultBackendPath();
  bool backendPathSet = false;
  bool sawM = false, sawK = false, sawN = false;
  Shape explicitShape{0, 0, 0};

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    }
    if (arg == "--backend") {
      if (++i >= argc) throw std::runtime_error("--backend requires a value");
      options.backendPath = argv[i];
      backendPathSet      = true;
      continue;
    }
    if (arg == "--m") {
      if (++i >= argc) throw std::runtime_error("--m requires a value");
      explicitShape.rows = parseUintOption(argv[i], "--m");
      sawM = true;
      continue;
    }
    if (arg == "--k") {
      if (++i >= argc) throw std::runtime_error("--k requires a value");
      explicitShape.inner = parseUintOption(argv[i], "--k");
      sawK = true;
      continue;
    }
    if (arg == "--n") {
      if (++i >= argc) throw std::runtime_error("--n requires a value");
      explicitShape.cols = parseUintOption(argv[i], "--n");
      sawN = true;
      continue;
    }
    if (arg == "--num-weights") {
      if (++i >= argc) throw std::runtime_error("--num-weights requires a value");
      options.numWeights = parseUintOption(argv[i], "--num-weights");
      continue;
    }
    if (arg == "--warmup") {
      if (++i >= argc) throw std::runtime_error("--warmup requires a value");
      options.warmup = parseUintOption(argv[i], "--warmup");
      continue;
    }
    if (arg == "--iters") {
      if (++i >= argc) throw std::runtime_error("--iters requires a value");
      options.iterations = parseUintOption(argv[i], "--iters");
      continue;
    }
    if (arg == "--precisions") {
      if (++i >= argc) throw std::runtime_error("--precisions requires a value");
      options.precisions = parsePrecisionList(argv[i]);
      continue;
    }
    if (arg == "--log") {
      if (++i >= argc) throw std::runtime_error("--log requires a value");
      options.logPath = argv[i];
      continue;
    }
    if (arg.rfind("--", 0) == 0) throw std::runtime_error("unknown option: " + arg);
    if (backendPathSet) throw std::runtime_error("multiple backend paths were provided");
    options.backendPath = arg;
    backendPathSet      = true;
  }

  if (sawM || sawK || sawN) {
    if (!(sawM && sawK && sawN)) throw std::runtime_error("--m/--k/--n must be provided together");
    options.shapes = {explicitShape};
  }
  if (options.numWeights == 0) throw std::runtime_error("--num-weights must be non-zero");
  if (options.iterations == 0) throw std::runtime_error("--iters must be non-zero");
  if (options.precisions.empty()) throw std::runtime_error("--precisions must not be empty");
  return options;
}

BackendKind detectBackendKind(const std::string& backendPath) {
  const std::string path = toLower(backendPath);
  if (path.find("qnnhtp") != std::string::npos) return BackendKind::Htp;
  if (path.find("qnngpu") != std::string::npos) return BackendKind::Gpu;
  if (path.find("qnncpu") != std::string::npos) return BackendKind::Cpu;
  return BackendKind::Unknown;
}

const char* backendKindName(BackendKind kind) {
  switch (kind) {
    case BackendKind::Cpu: return "CPU";
    case BackendKind::Gpu: return "GPU";
    case BackendKind::Htp: return "NPU";
    case BackendKind::Unknown: return "UNKNOWN";
  }
  return "UNKNOWN";
}

QnnLog_Level_t parseLogLevelFromEnv() {
  const char* raw = std::getenv("QNN_LOG_LEVEL");
  if (raw == nullptr || *raw == '\0') return QNN_LOG_LEVEL_ERROR;
  const std::string value = toLower(raw);
  if (value == "warn") return QNN_LOG_LEVEL_WARN;
  if (value == "info") return QNN_LOG_LEVEL_INFO;
  if (value == "verbose") return QNN_LOG_LEVEL_VERBOSE;
  if (value == "debug") return QNN_LOG_LEVEL_DEBUG;
  return QNN_LOG_LEVEL_ERROR;
}

const char* logLevelName(QnnLog_Level_t level) {
  switch (level) {
    case QNN_LOG_LEVEL_ERROR: return "ERROR";
    case QNN_LOG_LEVEL_WARN: return "WARN";
    case QNN_LOG_LEVEL_INFO: return "INFO";
    case QNN_LOG_LEVEL_VERBOSE: return "VERBOSE";
    case QNN_LOG_LEVEL_DEBUG: return "DEBUG";
    case QNN_LOG_LEVEL_MAX: return "MAX";
  }
  return "UNKNOWN";
}

void qnnLogCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
  std::array<char, 2048> buffer{};
  va_list argsCopy;
  va_copy(argsCopy, args);
  std::vsnprintf(buffer.data(), buffer.size(), fmt, argsCopy);
  va_end(argsCopy);
  std::cerr << "[QNN][" << logLevelName(level) << "][" << timestamp << "] " << buffer.data()
            << '\n';
}

uint32_t defaultHtpSocModel() { return QNN_SOC_MODEL_SM8750; }
QnnHtpDevice_Arch_t defaultHtpArch() { return QNN_HTP_DEVICE_ARCH_V79; }

std::string getErrorText(const QNN_INTERFACE_VER_TYPE& qnnInterface, Qnn_ErrorHandle_t error) {
  if (qnnInterface.errorGetMessage != nullptr) {
    const char* message = nullptr;
    if (qnnInterface.errorGetMessage(error, &message) == QNN_SUCCESS && message != nullptr) {
      return message;
    }
  }
  return "QNN error code " + std::to_string(static_cast<uint64_t>(error));
}

void checkQnnStatus(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                    Qnn_ErrorHandle_t status,
                    const std::string& stage) {
  if (status != QNN_SUCCESS) throw std::runtime_error(stage + " failed: " + getErrorText(qnnInterface, status));
}

double elapsedMilliseconds(Clock::time_point start, Clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

double measureMilliseconds(const std::function<void()>& fn) {
  const auto start = Clock::now();
  fn();
  return elapsedMilliseconds(start, Clock::now());
}

double computeSampleStddev(const std::vector<double>& samples, double mean) {
  if (samples.size() <= 1) return 0.0;
  double sumSquaredDiff = 0.0;
  for (double sample : samples) {
    const double diff = sample - mean;
    sumSquaredDiff += diff * diff;
  }
  return std::sqrt(sumSquaredDiff / static_cast<double>(samples.size() - 1));
}

std::string formatMeanStd(double mean, double stddev) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << mean << "±" << stddev;
  return stream.str();
}

double computeMean(const std::vector<double>& samples) {
  if (samples.empty()) return 0.0;
  return std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
}

std::string shapeTag(Shape shape) {
  return std::to_string(shape.rows) + "x" + std::to_string(shape.inner) + "x" +
         std::to_string(shape.cols);
}

std::string makeArtifactStem(const std::string& mode, PrecisionKind precision, Shape shape) {
  return "weight_switch_" + mode + "_" + precisionKindName(precision) + "_" + shapeTag(shape);
}

uint32_t bytesPerElement(Qnn_DataType_t dataType) {
  switch (dataType) {
    case QNN_DATATYPE_FLOAT_32: return sizeof(float);
    case QNN_DATATYPE_FLOAT_16: return sizeof(uint16_t);
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8: return sizeof(uint8_t);
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: return sizeof(uint16_t);
    default: throw std::runtime_error("unsupported tensor data type size");
  }
}

size_t tensorStorageBytes(const TensorBuffer& buffer) {
  return buffer.elements * bytesPerElement(buffer.dataType);
}

uint32_t tensorDataSize(const TensorBuffer& buffer) {
  const size_t size = buffer.storage.empty() ? tensorStorageBytes(buffer) : buffer.storage.size();
  return static_cast<uint32_t>(size);
}

// The offline-style cases intentionally materialize tensors and binaries on disk so the benchmark
// can separate filesystem read time from backend load/update time.
void writeBytesToFile(const std::string& path, const void* data, size_t size) {
  std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("failed to open file for write: " + path);
  if (size != 0) out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!out) throw std::runtime_error("failed to write file: " + path);
}

std::vector<uint8_t> readBytesFromFile(const std::string& path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) throw std::runtime_error("failed to open file for read: " + path);
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) throw std::runtime_error("failed to get file size: " + path);
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (!bytes.empty()) {
    in.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!in) throw std::runtime_error("failed to read file: " + path);
  }
  return bytes;
}

void writeBufferToFile(const std::string& path, const TensorBuffer& buffer) {
  writeBytesToFile(path, buffer.data(), buffer.storage.size());
}

TensorBuffer loadBufferFromFile(const std::string& path, const TensorBuffer& prototype) {
  TensorBuffer loaded = prototype;
  loaded.storage      = readBytesFromFile(path);
  const size_t expectedBytes = tensorStorageBytes(prototype);
  if (loaded.storage.size() != expectedBytes) {
    throw std::runtime_error("tensor file size mismatch for " + path);
  }
  loaded.refreshQuantizationPointers();
  return loaded;
}

void releaseTensorStorage(TensorBuffer& buffer) {
  std::vector<uint8_t>().swap(buffer.storage);
}

std::vector<float> readFloatVectorFromFile(const std::string& path, size_t elements) {
  std::vector<uint8_t> bytes = readBytesFromFile(path);
  const size_t expectedBytes = elements * sizeof(float);
  if (bytes.size() != expectedBytes) {
    throw std::runtime_error("float vector file size mismatch for " + path);
  }
  std::vector<float> values(elements, 0.0f);
  if (!values.empty()) std::memcpy(values.data(), bytes.data(), expectedBytes);
  return values;
}

TensorFileArtifacts writeTensorArtifacts(const std::string& stem,
                                         const TensorBuffer& inputBuffer,
                                         const std::vector<TensorBuffer>& weightBuffers) {
  TensorFileArtifacts artifacts{};
  artifacts.inputPath = stem + "_input.bin";
  writeBufferToFile(artifacts.inputPath, inputBuffer);
  artifacts.weightPaths.reserve(weightBuffers.size());
  for (size_t i = 0; i < weightBuffers.size(); ++i) {
    const std::string path = stem + "_weight_" + std::to_string(i) + ".bin";
    writeBufferToFile(path, weightBuffers[i]);
    artifacts.weightPaths.push_back(path);
  }
  return artifacts;
}

uint16_t floatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  uint32_t mantissa   = bits & 0x007FFFFFu;
  int32_t exponent    = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x00800000u) >> static_cast<uint32_t>(1 - exponent);
    if ((mantissa & 0x00001000u) != 0u) mantissa += 0x00002000u;
    return static_cast<uint16_t>(sign | (mantissa >> 13));
  }
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  if ((mantissa & 0x00001000u) != 0u) {
    mantissa += 0x00002000u;
    if ((mantissa & 0x00800000u) != 0u) {
      mantissa = 0;
      ++exponent;
      if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               (mantissa >> 13));
}

float halfBitsToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  uint32_t exponent   = (bits >> 10) & 0x1Fu;
  uint32_t mantissa   = bits & 0x03FFu;
  uint32_t floatBits  = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      floatBits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0u) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03FFu;
      exponent = exponent + (127 - 15);
      floatBits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    floatBits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    exponent  = exponent + (127 - 15);
    floatBits = sign | (exponent << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &floatBits, sizeof(value));
  return value;
}

std::vector<float> makeData(size_t size, int modulo, float scale, uint32_t salt = 0) {
  std::vector<float> values(size);
  for (size_t i = 0; i < size; ++i) {
    const int value = static_cast<int>((i + salt * 13) % static_cast<size_t>(modulo));
    values[i]       = static_cast<float>(value - (modulo / 2)) / scale;
  }
  return values;
}

template <typename T>
void writeTypedStorage(TensorBuffer& buffer, const std::vector<T>& values) {
  buffer.elements = values.size();
  buffer.storage.resize(values.size() * sizeof(T));
  if (!values.empty()) std::memcpy(buffer.storage.data(), values.data(), buffer.storage.size());
}

TensorBuffer makeFloat32Buffer(const std::vector<float>& values) {
  TensorBuffer buffer{};
  buffer.dataType = QNN_DATATYPE_FLOAT_32;
  writeTypedStorage(buffer, values);
  return buffer;
}

TensorBuffer makeFp16Buffer(const std::vector<float>& values) {
  TensorBuffer buffer{};
  buffer.dataType = QNN_DATATYPE_FLOAT_16;
  std::vector<uint16_t> bits(values.size(), 0u);
  for (size_t i = 0; i < values.size(); ++i) bits[i] = floatToHalfBits(values[i]);
  writeTypedStorage(buffer, bits);
  return buffer;
}

TensorBuffer makeSignedSymmetricBuffer(const std::vector<float>& values,
                                       Qnn_DataType_t dataType,
                                       int bits) {
  TensorBuffer buffer{};
  buffer.dataType                            = dataType;
  buffer.quantizeParams.encodingDefinition   = QNN_DEFINITION_DEFINED;
  buffer.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;

  float maxAbs = 0.0f;
  for (float value : values) maxAbs = std::max(maxAbs, std::abs(value));
  const int64_t qmax = (static_cast<int64_t>(1) << (bits - 1)) - 1;
  const double scale =
      (maxAbs <= std::numeric_limits<float>::epsilon()) ? 1.0 : (maxAbs / qmax);
  buffer.quantizeParams.scaleOffsetEncoding.scale  = static_cast<float>(scale);
  buffer.quantizeParams.scaleOffsetEncoding.offset = 0;

  if (bits == 8) {
    std::vector<int8_t> quantized(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto q       = static_cast<int64_t>(std::llround(values[i] / scale));
      const auto clamped = std::clamp<int64_t>(q, -qmax, qmax);
      quantized[i]       = static_cast<int8_t>(clamped);
    }
    writeTypedStorage(buffer, quantized);
    return buffer;
  }

  if (bits == 16) {
    std::vector<int16_t> quantized(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto q       = static_cast<int64_t>(std::llround(values[i] / scale));
      const auto clamped = std::clamp<int64_t>(q, -qmax, qmax);
      quantized[i]       = static_cast<int16_t>(clamped);
    }
    writeTypedStorage(buffer, quantized);
    return buffer;
  }

  throw std::runtime_error("unsupported signed symmetric bitwidth");
}

TensorBuffer makeSignedSymmetricBufferWithScale(const std::vector<float>& values,
                                                Qnn_DataType_t dataType,
                                                int bits,
                                                double scale) {
  TensorBuffer buffer{};
  buffer.dataType                            = dataType;
  buffer.quantizeParams.encodingDefinition   = QNN_DEFINITION_DEFINED;
  buffer.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  if (scale <= 0.0) scale = 1.0;
  buffer.quantizeParams.scaleOffsetEncoding.scale  = static_cast<float>(scale);
  buffer.quantizeParams.scaleOffsetEncoding.offset = 0;

  const int64_t qmax = (static_cast<int64_t>(1) << (bits - 1)) - 1;
  if (bits == 8) {
    std::vector<int8_t> quantized(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto q       = static_cast<int64_t>(std::llround(values[i] / scale));
      const auto clamped = std::clamp<int64_t>(q, -qmax, qmax);
      quantized[i]       = static_cast<int8_t>(clamped);
    }
    writeTypedStorage(buffer, quantized);
    return buffer;
  }

  if (bits == 16) {
    std::vector<int16_t> quantized(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto q       = static_cast<int64_t>(std::llround(values[i] / scale));
      const auto clamped = std::clamp<int64_t>(q, -qmax, qmax);
      quantized[i]       = static_cast<int16_t>(clamped);
    }
    writeTypedStorage(buffer, quantized);
    return buffer;
  }

  throw std::runtime_error("unsupported signed symmetric bitwidth");
}

TensorBuffer makeUnsignedAffineBuffer(const std::vector<float>& values,
                                      Qnn_DataType_t dataType,
                                      int bits) {
  TensorBuffer buffer{};
  buffer.dataType                            = dataType;
  buffer.quantizeParams.encodingDefinition   = QNN_DEFINITION_DEFINED;
  buffer.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;

  const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
  const double minValue     = (minIt == values.end()) ? 0.0 : *minIt;
  const double maxValue     = (maxIt == values.end()) ? 0.0 : *maxIt;
  const double qmax         = static_cast<double>((static_cast<uint64_t>(1) << bits) - 1);
  double scale              = (maxValue - minValue) / std::max(1.0, qmax);
  if (scale <= 1e-9) scale = 1.0;
  const double zeroPoint =
      std::clamp(std::llround(-minValue / scale), 0ll, static_cast<long long>(qmax));
  buffer.quantizeParams.scaleOffsetEncoding.scale  = static_cast<float>(scale);
  buffer.quantizeParams.scaleOffsetEncoding.offset = -static_cast<int32_t>(zeroPoint);

  if (bits == 8) {
    std::vector<uint8_t> quantized(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto q = static_cast<long long>(
          std::llround(values[i] / scale) - buffer.quantizeParams.scaleOffsetEncoding.offset);
      quantized[i] = static_cast<uint8_t>(std::clamp<long long>(q, 0, 255));
    }
    writeTypedStorage(buffer, quantized);
    return buffer;
  }

  if (bits == 16) {
    std::vector<uint16_t> quantized(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto q = static_cast<long long>(
          std::llround(values[i] / scale) - buffer.quantizeParams.scaleOffsetEncoding.offset);
      quantized[i] = static_cast<uint16_t>(std::clamp<long long>(q, 0, 65535));
    }
    writeTypedStorage(buffer, quantized);
    return buffer;
  }

  throw std::runtime_error("unsupported unsigned affine bitwidth");
}

TensorBuffer makeUnsignedAffineBufferWithParams(const std::vector<float>& values,
                                                Qnn_DataType_t dataType,
                                                int bits,
                                                double scale,
                                                int32_t offset) {
  TensorBuffer buffer{};
  buffer.dataType                            = dataType;
  buffer.quantizeParams.encodingDefinition   = QNN_DEFINITION_DEFINED;
  buffer.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  if (scale <= 0.0) scale = 1.0;
  buffer.quantizeParams.scaleOffsetEncoding.scale  = static_cast<float>(scale);
  buffer.quantizeParams.scaleOffsetEncoding.offset = offset;

  if (bits == 8) {
    std::vector<uint8_t> quantized(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto q = static_cast<long long>(std::llround(values[i] / scale) - offset);
      quantized[i] = static_cast<uint8_t>(std::clamp<long long>(q, 0, 255));
    }
    writeTypedStorage(buffer, quantized);
    return buffer;
  }

  if (bits == 16) {
    std::vector<uint16_t> quantized(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto q = static_cast<long long>(std::llround(values[i] / scale) - offset);
      quantized[i] = static_cast<uint16_t>(std::clamp<long long>(q, 0, 65535));
    }
    writeTypedStorage(buffer, quantized);
    return buffer;
  }

  throw std::runtime_error("unsupported unsigned affine bitwidth");
}

TensorBuffer makeSignedInt8AxisWeightBuffer(const std::vector<float>& values,
                                            uint32_t rows,
                                            uint32_t cols) {
  TensorBuffer buffer{};
  buffer.dataType                            = QNN_DATATYPE_SFIXED_POINT_8;
  buffer.quantizeParams.encodingDefinition   = QNN_DEFINITION_DEFINED;
  buffer.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  buffer.quantizeParams.axisScaleOffsetEncoding.axis = 1;
  buffer.axisScaleOffsets.resize(cols, QNN_SCALE_OFFSET_INIT);

  std::vector<int8_t> quantized(values.size(), 0);
  constexpr int64_t kQmax = 127;
  for (uint32_t col = 0; col < cols; ++col) {
    float maxAbs = 0.0f;
    for (uint32_t row = 0; row < rows; ++row) {
      maxAbs = std::max(maxAbs, std::abs(values[row * cols + col]));
    }
    const double scale =
        (maxAbs <= std::numeric_limits<float>::epsilon()) ? 1.0 : (maxAbs / kQmax);
    buffer.axisScaleOffsets[col].scale  = static_cast<float>(scale);
    buffer.axisScaleOffsets[col].offset = 0;
    for (uint32_t row = 0; row < rows; ++row) {
      const float value  = values[row * cols + col];
      const auto q       = static_cast<int64_t>(std::llround(value / scale));
      const auto clamped = std::clamp<int64_t>(q, -kQmax, kQmax);
      quantized[row * cols + col] = static_cast<int8_t>(clamped);
    }
  }
  writeTypedStorage(buffer, quantized);
  buffer.refreshQuantizationPointers();
  return buffer;
}

std::vector<Qnn_ScaleOffset_t> makeCommonSignedInt8AxisScales(
    const std::vector<std::vector<float>>& weights,
    uint32_t rows,
    uint32_t cols) {
  std::vector<Qnn_ScaleOffset_t> scales(cols, QNN_SCALE_OFFSET_INIT);
  constexpr int64_t kQmax = 127;
  for (uint32_t col = 0; col < cols; ++col) {
    float maxAbs = 0.0f;
    for (const auto& weight : weights) {
      for (uint32_t row = 0; row < rows; ++row) {
        maxAbs = std::max(maxAbs, std::abs(weight[row * cols + col]));
      }
    }
    const double scale =
        (maxAbs <= std::numeric_limits<float>::epsilon()) ? 1.0 : (maxAbs / kQmax);
    scales[col].scale  = static_cast<float>(scale);
    scales[col].offset = 0;
  }
  return scales;
}

TensorBuffer makeSignedInt8AxisWeightBufferWithScales(
    const std::vector<float>& values,
    uint32_t rows,
    uint32_t cols,
    const std::vector<Qnn_ScaleOffset_t>& scales) {
  if (scales.size() != cols) throw std::runtime_error("int8 axis scale count mismatch");
  TensorBuffer buffer{};
  buffer.dataType                            = QNN_DATATYPE_SFIXED_POINT_8;
  buffer.quantizeParams.encodingDefinition   = QNN_DEFINITION_DEFINED;
  buffer.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  buffer.quantizeParams.axisScaleOffsetEncoding.axis = 1;
  buffer.axisScaleOffsets = scales;

  std::vector<int8_t> quantized(values.size(), 0);
  constexpr int64_t kQmax = 127;
  for (uint32_t col = 0; col < cols; ++col) {
    const double scale = std::max(static_cast<double>(scales[col].scale), 1e-12);
    for (uint32_t row = 0; row < rows; ++row) {
      const float value  = values[row * cols + col];
      const auto q       = static_cast<int64_t>(std::llround(value / scale));
      const auto clamped = std::clamp<int64_t>(q, -kQmax, kQmax);
      quantized[row * cols + col] = static_cast<int8_t>(clamped);
    }
  }
  writeTypedStorage(buffer, quantized);
  buffer.refreshQuantizationPointers();
  return buffer;
}

void clearTensorStorage(TensorBuffer& buffer) {
  std::fill(buffer.storage.begin(), buffer.storage.end(), static_cast<uint8_t>(0));
}

void resizeTensorStorage(TensorBuffer& buffer, size_t elements) {
  buffer.elements = elements;
  buffer.storage.resize(elements * bytesPerElement(buffer.dataType));
}

std::vector<float> decodeTensorToFloat(const TensorBuffer& buffer) {
  std::vector<float> decoded(buffer.elements, 0.0f);
  switch (buffer.dataType) {
    case QNN_DATATYPE_FLOAT_32: {
      const auto* ptr = reinterpret_cast<const float*>(buffer.data());
      std::copy(ptr, ptr + buffer.elements, decoded.begin());
      return decoded;
    }
    case QNN_DATATYPE_FLOAT_16: {
      const auto* ptr = reinterpret_cast<const uint16_t*>(buffer.data());
      for (size_t i = 0; i < buffer.elements; ++i) decoded[i] = halfBitsToFloat(ptr[i]);
      return decoded;
    }
    case QNN_DATATYPE_SFIXED_POINT_8: {
      const auto* ptr       = reinterpret_cast<const int8_t*>(buffer.data());
      const float scale     = buffer.quantizeParams.scaleOffsetEncoding.scale;
      const int32_t offset  = buffer.quantizeParams.scaleOffsetEncoding.offset;
      for (size_t i = 0; i < buffer.elements; ++i) {
        decoded[i] = (static_cast<int32_t>(ptr[i]) + offset) * scale;
      }
      return decoded;
    }
    case QNN_DATATYPE_UFIXED_POINT_8: {
      const auto* ptr       = reinterpret_cast<const uint8_t*>(buffer.data());
      const float scale     = buffer.quantizeParams.scaleOffsetEncoding.scale;
      const int32_t offset  = buffer.quantizeParams.scaleOffsetEncoding.offset;
      for (size_t i = 0; i < buffer.elements; ++i) {
        decoded[i] = (static_cast<int32_t>(ptr[i]) + offset) * scale;
      }
      return decoded;
    }
    case QNN_DATATYPE_SFIXED_POINT_16: {
      const auto* ptr       = reinterpret_cast<const int16_t*>(buffer.data());
      const float scale     = buffer.quantizeParams.scaleOffsetEncoding.scale;
      const int32_t offset  = buffer.quantizeParams.scaleOffsetEncoding.offset;
      for (size_t i = 0; i < buffer.elements; ++i) {
        decoded[i] = (static_cast<int32_t>(ptr[i]) + offset) * scale;
      }
      return decoded;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      const auto* ptr       = reinterpret_cast<const uint16_t*>(buffer.data());
      const float scale     = buffer.quantizeParams.scaleOffsetEncoding.scale;
      const int32_t offset  = buffer.quantizeParams.scaleOffsetEncoding.offset;
      for (size_t i = 0; i < buffer.elements; ++i) {
        decoded[i] = (static_cast<int32_t>(ptr[i]) + offset) * scale;
      }
      return decoded;
    }
    default:
      throw std::runtime_error("unsupported output data type for decode");
  }
}

std::vector<float> referenceMatmul(const std::vector<float>& input,
                                   const std::vector<float>& weights,
                                   Shape shape) {
  std::vector<float> output(static_cast<size_t>(shape.rows) * shape.cols, 0.0f);
  for (uint32_t row = 0; row < shape.rows; ++row) {
    for (uint32_t col = 0; col < shape.cols; ++col) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < shape.inner; ++k) {
        acc += input[row * shape.inner + k] * weights[k * shape.cols + col];
      }
      output[row * shape.cols + col] = acc;
    }
  }
  return output;
}

void validateOutput(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    double tolerance,
                    CaseResult& result) {
  if (actual.size() != expected.size()) throw std::runtime_error("output size mismatch");
  double maxAbsError = 0.0;
  double sumAbsError = 0.0;
  bool invalid       = false;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i])) {
      invalid = true;
      break;
    }
    const double absError =
        std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
    maxAbsError = std::max(maxAbsError, absError);
    sumAbsError += absError;
  }
  result.maxAbsError     = invalid ? std::numeric_limits<double>::infinity() : maxAbsError;
  result.meanAbsError    = invalid ? std::numeric_limits<double>::infinity()
                                   : sumAbsError / static_cast<double>(actual.size());
  result.validationPassed = !invalid && result.maxAbsError <= tolerance;
}

double validationToleranceForCase(BackendKind kind, PrecisionKind precision) {
  if (precision == PrecisionKind::Int8) return 5e-2;
  if (precision == PrecisionKind::Int16) return 2e-2;
  return kind == BackendKind::Htp ? 5e-3 : 1e-4;
}

void prepareBuffers(BackendKind backendKind,
                    PrecisionKind precision,
                    uint32_t inner,
                    uint32_t cols,
                    const std::vector<float>& inputDataFloat,
                    const std::vector<float>& weightDataFloat,
                    const std::vector<float>& expectedOutputFloat,
                    TensorBuffer& inputBuffer,
                    TensorBuffer& weightBuffer,
                    TensorBuffer& outputBuffer) {
  switch (precision) {
    case PrecisionKind::Fp32:
      inputBuffer  = makeFloat32Buffer(inputDataFloat);
      weightBuffer = makeFloat32Buffer(weightDataFloat);
      outputBuffer = makeFloat32Buffer(expectedOutputFloat);
      clearTensorStorage(outputBuffer);
      return;
    case PrecisionKind::Fp16:
      inputBuffer  = makeFp16Buffer(inputDataFloat);
      weightBuffer = makeFp16Buffer(weightDataFloat);
      outputBuffer = makeFp16Buffer(expectedOutputFloat);
      clearTensorStorage(outputBuffer);
      return;
    case PrecisionKind::Int8:
      if (backendKind == BackendKind::Cpu) {
        inputBuffer  = makeUnsignedAffineBuffer(inputDataFloat, QNN_DATATYPE_UFIXED_POINT_8, 8);
        weightBuffer = makeUnsignedAffineBuffer(weightDataFloat, QNN_DATATYPE_UFIXED_POINT_8, 8);
        outputBuffer =
            makeUnsignedAffineBuffer(expectedOutputFloat, QNN_DATATYPE_UFIXED_POINT_8, 8);
        clearTensorStorage(outputBuffer);
        return;
      }
      inputBuffer  = makeSignedSymmetricBuffer(inputDataFloat, QNN_DATATYPE_SFIXED_POINT_8, 8);
      weightBuffer = makeSignedInt8AxisWeightBuffer(weightDataFloat, inner, cols);
      outputBuffer =
          makeSignedSymmetricBuffer(expectedOutputFloat, QNN_DATATYPE_SFIXED_POINT_8, 8);
      clearTensorStorage(outputBuffer);
      return;
    case PrecisionKind::Int16:
      inputBuffer  = makeSignedSymmetricBuffer(inputDataFloat, QNN_DATATYPE_SFIXED_POINT_16, 16);
      weightBuffer = makeSignedSymmetricBuffer(weightDataFloat, QNN_DATATYPE_SFIXED_POINT_16, 16);
      outputBuffer =
          makeSignedSymmetricBuffer(expectedOutputFloat, QNN_DATATYPE_SFIXED_POINT_16, 16);
      clearTensorStorage(outputBuffer);
      return;
  }
}

Qnn_Tensor_t makeTensor(const char* name,
                        Qnn_TensorType_t type,
                        Qnn_DataType_t dataType,
                        const Qnn_QuantizeParams_t& quantizeParams,
                        uint32_t* dimensions,
                        uint32_t rank,
                        void* data,
                        uint32_t dataSize) {
  Qnn_Tensor_t tensor          = QNN_TENSOR_INIT;
  tensor.version               = QNN_TENSOR_VERSION_1;
  tensor.v1.name               = name;
  tensor.v1.type               = type;
  tensor.v1.dataFormat         = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor.v1.dataType           = dataType;
  tensor.v1.quantizeParams     = quantizeParams;
  tensor.v1.rank               = rank;
  tensor.v1.dimensions         = dimensions;
  tensor.v1.memType            = QNN_TENSORMEMTYPE_RAW;
  tensor.v1.clientBuf.data     = data;
  tensor.v1.clientBuf.dataSize = dataSize;
  return tensor;
}

Qnn_Tensor_t makeExecuteTensor(const Qnn_Tensor_t& registeredTensor, void* data, uint32_t dataSize) {
  Qnn_Tensor_t tensor          = registeredTensor;
  tensor.v1.clientBuf.data     = data;
  tensor.v1.clientBuf.dataSize = dataSize;
  return tensor;
}

void addMatmulNode(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                   Qnn_GraphHandle_t graph,
                   Qnn_Tensor_t inputTensor,
                   Qnn_Tensor_t weightTensor,
                   Qnn_Tensor_t outputTensor) {
  std::array<Qnn_Tensor_t, 2> nodeInputs  = {inputTensor, weightTensor};
  std::array<Qnn_Tensor_t, 1> nodeOutputs = {outputTensor};
  Qnn_OpConfig_t opConfig                 = QNN_OPCONFIG_INIT;
  opConfig.v1.name                        = "matmul_0";
  opConfig.v1.packageName                 = QNN_OP_PACKAGE_NAME_QTI_AISW;
  opConfig.v1.typeName                    = QNN_OP_MAT_MUL;
  opConfig.v1.numOfInputs                 = static_cast<uint32_t>(nodeInputs.size());
  opConfig.v1.inputTensors                = nodeInputs.data();
  opConfig.v1.numOfOutputs                = static_cast<uint32_t>(nodeOutputs.size());
  opConfig.v1.outputTensors               = nodeOutputs.data();
  checkQnnStatus(qnnInterface, qnnInterface.graphAddNode(graph, opConfig), "QnnGraph_addNode");
}

void createDevice(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                  Qnn_LogHandle_t logger,
                  BackendKind backendKind,
                  Qnn_DeviceHandle_t* device,
                  const QnnDevice_PlatformInfo_t** platformInfo) {
  if (qnnInterface.deviceCreate == nullptr) {
    *device = nullptr;
    return;
  }
  auto tryDeviceCreate = [&](const QnnDevice_Config_t** configs) {
    *device = nullptr;
    return qnnInterface.deviceCreate(logger, configs, device);
  };
  auto status = tryDeviceCreate(nullptr);
  if (status == QNN_SUCCESS || status == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
    if (status == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) *device = nullptr;
    return;
  }
  if (backendKind == BackendKind::Htp) {
    if (qnnInterface.deviceGetPlatformInfo != nullptr) {
      const auto infoStatus = qnnInterface.deviceGetPlatformInfo(logger, platformInfo);
      if (infoStatus == QNN_SUCCESS && *platformInfo != nullptr) {
        QnnDevice_Config_t platformConfig = QNN_DEVICE_CONFIG_INIT;
        platformConfig.option             = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
        platformConfig.hardwareInfo       = const_cast<QnnDevice_PlatformInfo_t*>(*platformInfo);
        const QnnDevice_Config_t* platformConfigs[] = {&platformConfig, nullptr};
        status = tryDeviceCreate(platformConfigs);
      }
    }
    if (status != QNN_SUCCESS) {
      QnnHtpDevice_CustomConfig_t htpArchConfig{};
      QnnDevice_Config_t htpDeviceConfig = QNN_DEVICE_CONFIG_INIT;
      htpArchConfig.option        = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
      htpArchConfig.arch.deviceId = 0;
      htpArchConfig.arch.arch     = defaultHtpArch();
      htpDeviceConfig.option       = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      htpDeviceConfig.customConfig = &htpArchConfig;
      const QnnDevice_Config_t* htpDeviceConfigs[] = {&htpDeviceConfig, nullptr};
      status = tryDeviceCreate(htpDeviceConfigs);
    }
    if (status != QNN_SUCCESS) {
      QnnHtpDevice_CustomConfig_t htpSocConfig{};
      QnnDevice_Config_t htpDeviceConfig = QNN_DEVICE_CONFIG_INIT;
      htpSocConfig.option   = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
      htpSocConfig.socModel = defaultHtpSocModel();
      htpDeviceConfig.option       = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      htpDeviceConfig.customConfig = &htpSocConfig;
      const QnnDevice_Config_t* htpDeviceConfigs[] = {&htpDeviceConfig, nullptr};
      status = tryDeviceCreate(htpDeviceConfigs);
    }
  }
  if (status != QNN_SUCCESS && status != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
    throw std::runtime_error("QnnDevice_create failed: " + getErrorText(qnnInterface, status));
  }
  if (status == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) *device = nullptr;
}

std::vector<uint8_t> getContextBinary(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                                      Qnn_ContextHandle_t context) {
  if (qnnInterface.contextGetBinarySize == nullptr || qnnInterface.contextGetBinary == nullptr) {
    throw std::runtime_error("context binary APIs are unavailable");
  }
  Qnn_ContextBinarySize_t binarySize = 0;
  checkQnnStatus(qnnInterface,
                 qnnInterface.contextGetBinarySize(context, &binarySize),
                 "QnnContext_getBinarySize");
  std::vector<uint8_t> binary(static_cast<size_t>(binarySize));
  Qnn_ContextBinarySize_t writtenSize = 0;
  checkQnnStatus(qnnInterface,
                 qnnInterface.contextGetBinary(context, binary.data(), binarySize, &writtenSize),
                 "QnnContext_getBinary");
  binary.resize(static_cast<size_t>(writtenSize));
  return binary;
}

bool tryGetContextBinarySize(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                             Qnn_ContextHandle_t context,
                             uint64_t& binaryBytes) {
  binaryBytes = 0;
  if (qnnInterface.contextGetBinarySize == nullptr) return false;
  Qnn_ContextBinarySize_t binarySize = 0;
  const Qnn_ErrorHandle_t status = qnnInterface.contextGetBinarySize(context, &binarySize);
  if (status != QNN_SUCCESS) return false;
  binaryBytes = static_cast<uint64_t>(binarySize);
  return true;
}

StaticBinary compileStaticBinary(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                                 Qnn_BackendHandle_t backend,
                                 Qnn_DeviceHandle_t device,
                                 Shape shape,
                                 const TensorBuffer& inputBuffer,
                                 const TensorBuffer& weightBuffer,
                                 const TensorBuffer& outputBuffer,
                                 uint32_t graphIndex,
                                 const std::string& binaryFilePath) {
  // Build one graph whose weight is embedded as QNN_TENSOR_TYPE_STATIC, finalize it, and export
  // the whole context as a binary blob. The original context is freed after serialization.
  StaticBinary compiled{};
  compiled.graphName  = "static_weight_" + std::to_string(graphIndex);
  compiled.inputDims  = {shape.rows, shape.inner};
  std::array<uint32_t, 2> weightDims = {shape.inner, shape.cols};
  compiled.outputDims = {shape.rows, shape.cols};
  compiled.staticBytes = weightBuffer.bytes();
  compiled.binaryPath  = binaryFilePath;

  Qnn_ContextHandle_t context = nullptr;
  Qnn_GraphHandle_t graph     = nullptr;
  try {
    checkQnnStatus(qnnInterface,
                   qnnInterface.contextCreate(backend, device, nullptr, &context),
                   "QnnContext_create(static)");
    checkQnnStatus(qnnInterface,
                   qnnInterface.graphCreate(context, compiled.graphName.c_str(), nullptr, &graph),
                   "QnnGraph_create(static)");
    Qnn_Tensor_t inputTensor = makeTensor("input",
                                          QNN_TENSOR_TYPE_APP_WRITE,
                                          inputBuffer.dataType,
                                          inputBuffer.quantizeParams,
                                          compiled.inputDims.data(),
                                          2,
                                          nullptr,
                                          0);
    // Static mode gives QNN the actual weight bytes during tensor creation.
    Qnn_Tensor_t weightTensor = makeTensor("weight",
                                           QNN_TENSOR_TYPE_STATIC,
                                           weightBuffer.dataType,
                                           weightBuffer.quantizeParams,
                                           weightDims.data(),
                                           2,
                                           const_cast<void*>(weightBuffer.data()),
                                           weightBuffer.bytes());
    Qnn_Tensor_t outputTensor = makeTensor("output",
                                           QNN_TENSOR_TYPE_APP_READ,
                                           outputBuffer.dataType,
                                           outputBuffer.quantizeParams,
                                           compiled.outputDims.data(),
                                           2,
                                           nullptr,
                                           0);
    checkQnnStatus(qnnInterface,
                   qnnInterface.tensorCreateGraphTensor(graph, &inputTensor),
                   "QnnTensor_createGraphTensor(static input)");
    checkQnnStatus(qnnInterface,
                   qnnInterface.tensorCreateGraphTensor(graph, &weightTensor),
                   "QnnTensor_createGraphTensor(static weight)");
	    checkQnnStatus(qnnInterface,
	                   qnnInterface.tensorCreateGraphTensor(graph, &outputTensor),
	                   "QnnTensor_createGraphTensor(static output)");
	    addMatmulNode(qnnInterface, graph, inputTensor, weightTensor, outputTensor);
	    compiled.finalizeMs = measureMilliseconds([&]() {
	      checkQnnStatus(qnnInterface,
	                     qnnInterface.graphFinalize(graph, nullptr, nullptr),
	                     "QnnGraph_finalize(static)");
	    });
      // Persist the compiled context so runtime can measure filesystem load separately from
      // QnnContext_createFromBinary().
	    std::vector<uint8_t> binary = getContextBinary(qnnInterface, context);
      compiled.binaryBytes       = binary.size();
      writeBytesToFile(compiled.binaryPath, binary.data(), binary.size());
	    compiled.inputTensor  = inputTensor;
	    compiled.outputTensor = outputTensor;
	    checkQnnStatus(qnnInterface, qnnInterface.contextFree(context, nullptr), "QnnContext_free(static)");
	    context = nullptr;
	  } catch (...) {
    if (context != nullptr && qnnInterface.contextFree != nullptr) qnnInterface.contextFree(context, nullptr);
    throw;
  }
  return compiled;
}

struct ShapeData {
  TensorBuffer inputBuffer;
  TensorBuffer outputBuffer;
  std::vector<TensorBuffer> weightBuffers;
  TensorFileArtifacts artifacts;
  std::vector<std::string> expectedOutputPaths;
};

struct ShapeFloatData {
  std::vector<float> input;
  std::vector<std::vector<float>> weights;
  std::vector<std::vector<float>> expectedOutputs;
  std::vector<float> combinedWeights;
  std::vector<float> combinedExpectedOutputs;
};

ShapeFloatData makeShapeFloatData(const AppOptions& options, Shape shape) {
  // Generate fp32 source data and CPU reference outputs for one shape/precision preparation pass.
  // The data is deterministic, so regenerating it for each precision preserves comparability.
  ShapeFloatData data{};
  const size_t weightElements = static_cast<size_t>(shape.inner) * shape.cols;
  const size_t outputElements = static_cast<size_t>(shape.rows) * shape.cols;
  data.input = makeData(static_cast<size_t>(shape.rows) * shape.inner, 23, 11.0f);
  data.weights.reserve(options.numWeights);
  data.expectedOutputs.reserve(options.numWeights);
  data.combinedWeights.reserve(weightElements * options.numWeights);
  data.combinedExpectedOutputs.reserve(outputElements * options.numWeights);

  for (uint32_t i = 0; i < options.numWeights; ++i) {
    std::vector<float> weight =
        makeData(weightElements, 17 + static_cast<int>(i), 9.0f, i + 1);
    std::vector<float> expected = referenceMatmul(data.input, weight, shape);
    data.combinedWeights.insert(data.combinedWeights.end(), weight.begin(), weight.end());
    data.combinedExpectedOutputs.insert(
        data.combinedExpectedOutputs.end(), expected.begin(), expected.end());
    data.weights.push_back(std::move(weight));
    data.expectedOutputs.push_back(std::move(expected));
  }

  return data;
}

ShapeData makeShapeData(const AppOptions& options,
                        BackendKind backendKind,
                        PrecisionKind precision,
                        Shape shape,
                        const ShapeFloatData& floatData) {
  // Convert the fp32 source data into the precision and quantization encoding requested by QNN.
  // This host-side conversion is intentionally outside the timed execute() loop.
  ShapeData data{};
  const size_t outputElements = static_cast<size_t>(shape.rows) * shape.cols;
  data.weightBuffers.reserve(options.numWeights);

  switch (precision) {
    case PrecisionKind::Fp32:
      data.inputBuffer  = makeFloat32Buffer(floatData.input);
      data.outputBuffer = makeFloat32Buffer(floatData.combinedExpectedOutputs);
      for (const auto& weightFloat : floatData.weights) {
        data.weightBuffers.push_back(makeFloat32Buffer(weightFloat));
      }
      break;
    case PrecisionKind::Fp16:
      data.inputBuffer  = makeFp16Buffer(floatData.input);
      data.outputBuffer = makeFp16Buffer(floatData.combinedExpectedOutputs);
      for (const auto& weightFloat : floatData.weights) {
        data.weightBuffers.push_back(makeFp16Buffer(weightFloat));
      }
      break;
    case PrecisionKind::Int8:
      if (backendKind == BackendKind::Cpu) {
        // CPU uses unsigned affine int8 here.
        data.inputBuffer =
            makeUnsignedAffineBuffer(floatData.input, QNN_DATATYPE_UFIXED_POINT_8, 8);
        data.outputBuffer =
            makeUnsignedAffineBuffer(
                floatData.combinedExpectedOutputs, QNN_DATATYPE_UFIXED_POINT_8, 8);
        TensorBuffer commonWeight =
            makeUnsignedAffineBuffer(floatData.combinedWeights, QNN_DATATYPE_UFIXED_POINT_8, 8);
        for (const auto& weightFloat : floatData.weights) {
          data.weightBuffers.push_back(makeUnsignedAffineBufferWithParams(
              weightFloat,
              QNN_DATATYPE_UFIXED_POINT_8,
              8,
              commonWeight.quantizeParams.scaleOffsetEncoding.scale,
              commonWeight.quantizeParams.scaleOffsetEncoding.offset));
        }
      } else {
        // HTP MatMul uses signed symmetric activations and per-output-channel signed weights.
        data.inputBuffer =
            makeSignedSymmetricBuffer(floatData.input, QNN_DATATYPE_SFIXED_POINT_8, 8);
        data.outputBuffer = makeSignedSymmetricBuffer(
            floatData.combinedExpectedOutputs, QNN_DATATYPE_SFIXED_POINT_8, 8);
        const auto commonScales =
            makeCommonSignedInt8AxisScales(floatData.weights, shape.inner, shape.cols);
        for (const auto& weightFloat : floatData.weights) {
          data.weightBuffers.push_back(makeSignedInt8AxisWeightBufferWithScales(
              weightFloat, shape.inner, shape.cols, commonScales));
        }
      }
      break;
    case PrecisionKind::Int16: {
      data.inputBuffer =
          makeSignedSymmetricBuffer(floatData.input, QNN_DATATYPE_SFIXED_POINT_16, 16);
      data.outputBuffer =
          makeSignedSymmetricBuffer(
              floatData.combinedExpectedOutputs, QNN_DATATYPE_SFIXED_POINT_16, 16);
      TensorBuffer commonWeight =
          makeSignedSymmetricBuffer(floatData.combinedWeights, QNN_DATATYPE_SFIXED_POINT_16, 16);
      for (const auto& weightFloat : floatData.weights) {
        data.weightBuffers.push_back(makeSignedSymmetricBufferWithScale(
            weightFloat,
            QNN_DATATYPE_SFIXED_POINT_16,
            16,
            commonWeight.quantizeParams.scaleOffsetEncoding.scale));
      }
      break;
    }
  }

  data.inputBuffer.refreshQuantizationPointers();
  data.outputBuffer.refreshQuantizationPointers();
  for (auto& weightBuffer : data.weightBuffers) weightBuffer.refreshQuantizationPointers();

  const std::string stem = makeArtifactStem("tensor_store", precision, shape);
  data.artifacts = writeTensorArtifacts(stem, data.inputBuffer, data.weightBuffers);
  data.expectedOutputPaths.reserve(floatData.expectedOutputs.size());
  for (size_t i = 0; i < floatData.expectedOutputs.size(); ++i) {
    const std::string path = stem + "_expected_" + std::to_string(i) + ".f32";
    const auto& expected   = floatData.expectedOutputs[i];
    writeBytesToFile(path, expected.data(), expected.size() * sizeof(float));
    data.expectedOutputPaths.push_back(path);
  }

  releaseTensorStorage(data.inputBuffer);
  for (auto& weightBuffer : data.weightBuffers) releaseTensorStorage(weightBuffer);
  releaseTensorStorage(data.outputBuffer);
  resizeTensorStorage(data.outputBuffer, outputElements);
  clearTensorStorage(data.outputBuffer);
  return data;
}

CaseResult runAppWrite(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                       Qnn_BackendHandle_t backend,
                       Qnn_DeviceHandle_t device,
                       BackendKind backendKind,
                       const AppOptions& options,
                       PrecisionKind precision,
                       Shape shape,
                       ShapeData& data) {
  // One finalized graph handles all weights. Each execute() call binds the current weight buffer
  // as the second input tensor.
  CaseResult result{};
  result.shape = shape;
  result.precision = precision;
  result.mode = kModeAppWrite;
  result.runtimeBytes =
      static_cast<uint64_t>(tensorDataSize(data.inputBuffer)) + tensorDataSize(data.weightBuffers.front());

  Qnn_ContextHandle_t context = nullptr;
  Qnn_GraphHandle_t graph     = nullptr;
  std::array<uint32_t, 2> inputDims  = {shape.rows, shape.inner};
  std::array<uint32_t, 2> weightDims = {shape.inner, shape.cols};
  std::array<uint32_t, 2> outputDims = {shape.rows, shape.cols};
  Qnn_Tensor_t inputTensor  = QNN_TENSOR_INIT;
  Qnn_Tensor_t weightTensor = QNN_TENSOR_INIT;
  Qnn_Tensor_t outputTensor = QNN_TENSOR_INIT;

  try {
    result.prepareMs = measureMilliseconds([&]() {
      checkQnnStatus(qnnInterface,
                     qnnInterface.contextCreate(backend, device, nullptr, &context),
                     "QnnContext_create(app_write)");
      checkQnnStatus(qnnInterface,
                     qnnInterface.graphCreate(context, "app_write_weight", nullptr, &graph),
                     "QnnGraph_create(app_write)");
      inputTensor = makeTensor("input",
                               QNN_TENSOR_TYPE_APP_WRITE,
                               data.inputBuffer.dataType,
                               data.inputBuffer.quantizeParams,
                               inputDims.data(),
                               2,
                               nullptr,
                               0);
      // APP_WRITE weight registers only metadata at graph-build time; bytes arrive at execute().
      weightTensor = makeTensor("weight",
                                QNN_TENSOR_TYPE_APP_WRITE,
                                data.weightBuffers.front().dataType,
                                data.weightBuffers.front().quantizeParams,
                                weightDims.data(),
                                2,
                                nullptr,
                                0);
      outputTensor = makeTensor("output",
                                QNN_TENSOR_TYPE_APP_READ,
                                data.outputBuffer.dataType,
                                data.outputBuffer.quantizeParams,
                                outputDims.data(),
                                2,
                                nullptr,
                                0);
      checkQnnStatus(qnnInterface,
                     qnnInterface.tensorCreateGraphTensor(graph, &inputTensor),
                     "QnnTensor_createGraphTensor(app input)");
      checkQnnStatus(qnnInterface,
                     qnnInterface.tensorCreateGraphTensor(graph, &weightTensor),
                     "QnnTensor_createGraphTensor(app weight)");
      checkQnnStatus(qnnInterface,
                     qnnInterface.tensorCreateGraphTensor(graph, &outputTensor),
                     "QnnTensor_createGraphTensor(app output)");
      addMatmulNode(qnnInterface, graph, inputTensor, weightTensor, outputTensor);
    });

    result.finalizeMs = measureMilliseconds([&]() {
      checkQnnStatus(qnnInterface,
                     qnnInterface.graphFinalize(graph, nullptr, nullptr),
                     "QnnGraph_finalize(app_write)");
    });
    result.binaryBytes = getContextBinary(qnnInterface, context).size();
    uint32_t lastIndex = 0;
    std::vector<double> ssdDataSamples;
    std::vector<double> execSamples;
    ssdDataSamples.reserve(options.iterations);
    execSamples.reserve(options.iterations);

    auto executeOnce = [&](uint32_t iteration,
                           bool collectSample,
                           double* totalMs) {
      lastIndex = iteration % options.numWeights;
      const auto totalStart = Clock::now();
      TensorBuffer inputFromDisk;
      TensorBuffer weightFromDisk;
      // app_write keeps one finalized graph and changes only the runtime input payload.
      const double ssdDataMs = measureMilliseconds([&]() {
        inputFromDisk  = loadBufferFromFile(data.artifacts.inputPath, data.inputBuffer);
        weightFromDisk =
            loadBufferFromFile(data.artifacts.weightPaths[lastIndex], data.weightBuffers[lastIndex]);
      });
      Qnn_Tensor_t executeInput =
          makeExecuteTensor(inputTensor, inputFromDisk.data(), inputFromDisk.bytes());
      Qnn_Tensor_t executeOutput =
          makeExecuteTensor(outputTensor, data.outputBuffer.data(), data.outputBuffer.bytes());
      Qnn_Tensor_t executeWeight =
          makeExecuteTensor(weightTensor, weightFromDisk.data(), weightFromDisk.bytes());
      Qnn_Tensor_t executeInputs[]  = {executeInput, executeWeight};
      Qnn_Tensor_t executeOutputs[] = {executeOutput};
      const double execMs           = measureMilliseconds([&]() {
        checkQnnStatus(qnnInterface,
                       qnnInterface.graphExecute(
                           graph, executeInputs, 2, executeOutputs, 1, nullptr, nullptr),
                       "QnnGraph_execute(app_write)");
      });
      if (collectSample) {
        ssdDataSamples.push_back(ssdDataMs);
        execSamples.push_back(execMs);
      }
      if (totalMs != nullptr) *totalMs = elapsedMilliseconds(totalStart, Clock::now());
    };

    executeOnce(0, false, &result.firstMs);
    for (uint32_t i = 0; i < options.warmup; ++i) executeOnce(i, false, nullptr);
    for (uint32_t i = 0; i < options.iterations; ++i) {
      executeOnce(i, true, nullptr);
    }
    result.ssdDataLoadMs = computeMean(ssdDataSamples);
    result.avgMs         = computeMean(execSamples);
    result.stdMs         = computeSampleStddev(execSamples, result.avgMs);
    const std::vector<float> expectedOutput =
        readFloatVectorFromFile(data.expectedOutputPaths[lastIndex],
                                static_cast<size_t>(shape.rows) * shape.cols);
    validateOutput(decodeTensorToFloat(data.outputBuffer),
                   expectedOutput,
                   validationToleranceForCase(backendKind, precision),
                   result);

    result.releaseMs = measureMilliseconds([&]() {
      checkQnnStatus(qnnInterface, qnnInterface.contextFree(context, nullptr), "QnnContext_free(app)");
      context = nullptr;
    });
  } catch (const std::exception& ex) {
    result.status = "ERROR";
    result.detail = ex.what();
    if (context != nullptr && qnnInterface.contextFree != nullptr) qnnInterface.contextFree(context, nullptr);
  }
  return result;
}

CaseResult runStaticTempLoad(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                             Qnn_BackendHandle_t backend,
                             Qnn_DeviceHandle_t device,
                             BackendKind backendKind,
                             const AppOptions& options,
                             PrecisionKind precision,
                             Shape shape,
                             ShapeData& data) {
  // Each weight is compiled into its own static graph binary and written to disk. Every iteration
  // re-reads one binary, loads it into a temporary context, executes once, then frees the context.
  CaseResult result{};
  result.shape = shape;
  result.precision = precision;
  result.mode = kModeStaticLoad;
  result.runtimeBytes = tensorDataSize(data.inputBuffer);
  std::vector<StaticBinary> binaries;
  uint32_t lastIndex = 0;

  try {
    result.prepareMs = measureMilliseconds([&]() {
      const std::string stem = makeArtifactStem(kModeStaticLoad, precision, shape);
      binaries.reserve(options.numWeights);
      for (uint32_t i = 0; i < options.numWeights; ++i) {
        TensorBuffer weightFromDisk =
            loadBufferFromFile(data.artifacts.weightPaths[i], data.weightBuffers[i]);
        binaries.push_back(compileStaticBinary(qnnInterface,
                                               backend,
                                               device,
                                               shape,
                                               data.inputBuffer,
                                               weightFromDisk,
                                               data.outputBuffer,
                                               i,
                                               stem + "_graph_" + std::to_string(i) + ".bin"));
      }
    });
    for (const auto& binary : binaries) {
      result.binaryBytes += binary.binaryBytes;
      result.staticBytes += binary.staticBytes;
      result.finalizeMs += binary.finalizeMs;
    }
    result.prepareMs = std::max(0.0, result.prepareMs - result.finalizeMs);
    if (qnnInterface.contextCreateFromBinary == nullptr || qnnInterface.graphRetrieve == nullptr) {
      throw std::runtime_error("contextCreateFromBinary or graphRetrieve API is unavailable");
    }
    std::vector<double> ssdDataSamples;
    std::vector<double> ssdGraphSamples;
    std::vector<double> htpWriteSamples;
    std::vector<double> execSamples;
    ssdDataSamples.reserve(options.iterations);
    ssdGraphSamples.reserve(options.iterations);
    htpWriteSamples.reserve(options.iterations);
    execSamples.reserve(options.iterations);

    auto executeOnce = [&](uint32_t iteration, bool collectSample, double* totalMs) {
      lastIndex = iteration % options.numWeights;
      const auto totalStart = Clock::now();
      TensorBuffer inputFromDisk;
      std::vector<uint8_t> binaryBytes;
      const double ssdDataMs = measureMilliseconds([&]() {
        inputFromDisk = loadBufferFromFile(data.artifacts.inputPath, data.inputBuffer);
      });
      const double ssdGraphMs = measureMilliseconds([&]() {
        binaryBytes = readBytesFromFile(binaries[lastIndex].binaryPath);
      });
      LoadedGraph loaded{};
      // This stage measures backend-side work to deserialize the compiled graph and make it ready
      // for execution on HTP.
      const double htpWriteMs = measureMilliseconds([&]() {
        checkQnnStatus(qnnInterface,
                       qnnInterface.contextCreateFromBinary(backend,
                                                            device,
                                                            nullptr,
                                                            binaryBytes.data(),
                                                            binaryBytes.size(),
                                                            &loaded.context,
                                                            nullptr),
                       "QnnContext_createFromBinary");
        checkQnnStatus(qnnInterface,
                       qnnInterface.graphRetrieve(
                           loaded.context, binaries[lastIndex].graphName.c_str(), &loaded.graph),
                       "QnnGraph_retrieve");
        loaded.inputTensor  = binaries[lastIndex].inputTensor;
        loaded.outputTensor = binaries[lastIndex].outputTensor;
        loaded.inputTensor.v1.dimensions  = binaries[lastIndex].inputDims.data();
        loaded.outputTensor.v1.dimensions = binaries[lastIndex].outputDims.data();
      });
      Qnn_Tensor_t executeInput =
          makeExecuteTensor(loaded.inputTensor, inputFromDisk.data(), inputFromDisk.bytes());
      Qnn_Tensor_t executeOutput =
          makeExecuteTensor(loaded.outputTensor, data.outputBuffer.data(), data.outputBuffer.bytes());
      Qnn_Tensor_t executeInputs[]  = {executeInput};
      Qnn_Tensor_t executeOutputs[] = {executeOutput};
      const double execMs           = measureMilliseconds([&]() {
        checkQnnStatus(qnnInterface,
                       qnnInterface.graphExecute(
                           loaded.graph, executeInputs, 1, executeOutputs, 1, nullptr, nullptr),
                       "QnnGraph_execute(static temp)");
      });
      checkQnnStatus(
          qnnInterface, qnnInterface.contextFree(loaded.context, nullptr), "QnnContext_free(static temp)");
      loaded.context = nullptr;
      if (collectSample) {
        ssdDataSamples.push_back(ssdDataMs);
        ssdGraphSamples.push_back(ssdGraphMs);
        htpWriteSamples.push_back(htpWriteMs);
        execSamples.push_back(execMs);
      }
      if (totalMs != nullptr) *totalMs = elapsedMilliseconds(totalStart, Clock::now());
    };

    executeOnce(0, false, &result.firstMs);
    for (uint32_t i = 0; i < options.warmup; ++i) executeOnce(i, false, nullptr);
    for (uint32_t i = 0; i < options.iterations; ++i) {
      executeOnce(i, true, nullptr);
    }
    result.ssdDataLoadMs  = computeMean(ssdDataSamples);
    result.ssdGraphLoadMs = computeMean(ssdGraphSamples);
    result.htpWriteMs     = computeMean(htpWriteSamples);
    result.avgMs          = computeMean(execSamples);
    result.stdMs          = computeSampleStddev(execSamples, result.avgMs);
    const std::vector<float> expectedOutput =
        readFloatVectorFromFile(data.expectedOutputPaths[lastIndex],
                                static_cast<size_t>(shape.rows) * shape.cols);
    validateOutput(decodeTensorToFloat(data.outputBuffer),
                   expectedOutput,
                   validationToleranceForCase(backendKind, precision),
                   result);

    result.releaseMs = 0.0;
  } catch (const std::exception& ex) {
    result.status = "ERROR";
    result.detail = ex.what();
  }
  return result;
}

CaseResult runUpdateableStatic(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                               Qnn_BackendHandle_t backend,
                               Qnn_DeviceHandle_t device,
                               BackendKind backendKind,
                               const AppOptions& options,
                               PrecisionKind precision,
                               Shape shape,
                               ShapeData& data) {
  CaseResult result{};
  result.shape     = shape;
  result.precision = precision;
  result.mode      = kModeUpdateStatic;
  result.runtimeBytes = tensorDataSize(data.inputBuffer) + tensorDataSize(data.weightBuffers.front());

  if (backendKind == BackendKind::Htp) {
    result.status = "SKIP";
    result.detail =
        "HTP backend rejected UPDATEABLE_STATIC MatMul update/finalize path during device testing";
    return result;
  }

  if (qnnInterface.tensorUpdateGraphTensors == nullptr) {
    result.status = "ERROR";
    result.detail = "QnnTensor_updateGraphTensors API is unavailable";
    return result;
  }

  Qnn_ContextHandle_t context = nullptr;
  Qnn_GraphHandle_t graph     = nullptr;
  std::array<uint32_t, 2> inputDims  = {shape.rows, shape.inner};
  std::array<uint32_t, 2> weightDims = {shape.inner, shape.cols};
  std::array<uint32_t, 2> outputDims = {shape.rows, shape.cols};
  Qnn_Tensor_t inputTensor  = QNN_TENSOR_INIT;
  Qnn_Tensor_t weightTensor = QNN_TENSOR_INIT;
  Qnn_Tensor_t outputTensor = QNN_TENSOR_INIT;
  uint32_t lastIndex        = 0;

  try {
    TensorBuffer initialWeightFromDisk;
    result.prepareMs = measureMilliseconds([&]() {
      initialWeightFromDisk =
          loadBufferFromFile(data.artifacts.weightPaths.front(), data.weightBuffers.front());
      checkQnnStatus(qnnInterface,
                     qnnInterface.contextCreate(backend, device, nullptr, &context),
                     "QnnContext_create(updateable_static)");
      checkQnnStatus(qnnInterface,
                     qnnInterface.graphCreate(context, "updateable_static_weight", nullptr, &graph),
                     "QnnGraph_create(updateable_static)");
      inputTensor = makeTensor("input",
                               QNN_TENSOR_TYPE_APP_WRITE,
                               data.inputBuffer.dataType,
                               data.inputBuffer.quantizeParams,
                               inputDims.data(),
                               2,
                               nullptr,
                               0);
      weightTensor = makeTensor("weight",
                                QNN_TENSOR_TYPE_UPDATEABLE_STATIC,
                                data.weightBuffers.front().dataType,
                                data.weightBuffers.front().quantizeParams,
                                weightDims.data(),
                                2,
                                initialWeightFromDisk.data(),
                                initialWeightFromDisk.bytes());
      outputTensor = makeTensor("output",
                                QNN_TENSOR_TYPE_APP_READ,
                                data.outputBuffer.dataType,
                                data.outputBuffer.quantizeParams,
                                outputDims.data(),
                                2,
                                nullptr,
                                0);
      checkQnnStatus(qnnInterface,
                     qnnInterface.tensorCreateGraphTensor(graph, &inputTensor),
                     "QnnTensor_createGraphTensor(updateable input)");
      checkQnnStatus(qnnInterface,
                     qnnInterface.tensorCreateGraphTensor(graph, &weightTensor),
                     "QnnTensor_createGraphTensor(updateable weight)");
      checkQnnStatus(qnnInterface,
                     qnnInterface.tensorCreateGraphTensor(graph, &outputTensor),
                     "QnnTensor_createGraphTensor(updateable output)");
      addMatmulNode(qnnInterface, graph, inputTensor, weightTensor, outputTensor);
    });
    result.finalizeMs = measureMilliseconds([&]() {
      checkQnnStatus(qnnInterface,
                     qnnInterface.graphFinalize(graph, nullptr, nullptr),
                     "QnnGraph_finalize(updateable_static)");
    });
    uint64_t binaryBytes = 0;
    if (tryGetContextBinarySize(qnnInterface, context, binaryBytes)) {
      result.binaryBytes = binaryBytes;
    }
    result.staticBytes = tensorDataSize(data.weightBuffers.front());

    std::vector<double> ssdDataSamples;
    std::vector<double> htpWriteSamples;
    std::vector<double> execSamples;
    ssdDataSamples.reserve(options.iterations);
    htpWriteSamples.reserve(options.iterations);
    execSamples.reserve(options.iterations);

    auto executeOnce = [&](uint32_t iteration, bool collectSample, double* totalMs) {
      lastIndex = iteration % options.numWeights;
      const auto totalStart = Clock::now();
      TensorBuffer inputFromDisk;
      TensorBuffer weightFromDisk;
      const double ssdDataMs = measureMilliseconds([&]() {
        inputFromDisk  = loadBufferFromFile(data.artifacts.inputPath, data.inputBuffer);
        weightFromDisk =
            loadBufferFromFile(data.artifacts.weightPaths[lastIndex], data.weightBuffers[lastIndex]);
      });
      Qnn_Tensor_t updatedWeight = weightTensor;
      updatedWeight.v1.quantizeParams     = weightFromDisk.quantizeParams;
      updatedWeight.v1.clientBuf.data     = weightFromDisk.data();
      updatedWeight.v1.clientBuf.dataSize = weightFromDisk.bytes();
      const Qnn_Tensor_t* updates[]       = {&updatedWeight};
      // UPDATEABLE_STATIC does not change execute() inputs. The new weight becomes visible only
      // after updateGraphTensors() plus the follow-up finalize().
      const double htpWriteMs             = measureMilliseconds([&]() {
        checkQnnStatus(qnnInterface,
                       qnnInterface.tensorUpdateGraphTensors(graph, updates, 1),
                       "QnnTensor_updateGraphTensors");
        checkQnnStatus(qnnInterface,
                       qnnInterface.graphFinalize(graph, nullptr, nullptr),
                       "QnnGraph_finalize(updateable_static refresh)");
      });
      Qnn_Tensor_t executeInput =
          makeExecuteTensor(inputTensor, inputFromDisk.data(), inputFromDisk.bytes());
      Qnn_Tensor_t executeOutput =
          makeExecuteTensor(outputTensor, data.outputBuffer.data(), data.outputBuffer.bytes());
      Qnn_Tensor_t executeInputs[]  = {executeInput};
      Qnn_Tensor_t executeOutputs[] = {executeOutput};
      const double execMs           = measureMilliseconds([&]() {
        checkQnnStatus(qnnInterface,
                       qnnInterface.graphExecute(
                           graph, executeInputs, 1, executeOutputs, 1, nullptr, nullptr),
                       "QnnGraph_execute(updateable_static)");
      });
      if (collectSample) {
        ssdDataSamples.push_back(ssdDataMs);
        htpWriteSamples.push_back(htpWriteMs);
        execSamples.push_back(execMs);
      }
      if (totalMs != nullptr) *totalMs = elapsedMilliseconds(totalStart, Clock::now());
    };

    executeOnce(0, false, &result.firstMs);
    for (uint32_t i = 0; i < options.warmup; ++i) executeOnce(i, false, nullptr);
    for (uint32_t i = 0; i < options.iterations; ++i) {
      executeOnce(i, true, nullptr);
    }
    result.ssdDataLoadMs = computeMean(ssdDataSamples);
    result.htpWriteMs    = computeMean(htpWriteSamples);
    result.avgMs         = computeMean(execSamples);
    result.stdMs         = computeSampleStddev(execSamples, result.avgMs);
    const std::vector<float> expectedOutput =
        readFloatVectorFromFile(data.expectedOutputPaths[lastIndex],
                                static_cast<size_t>(shape.rows) * shape.cols);
    validateOutput(decodeTensorToFloat(data.outputBuffer),
                   expectedOutput,
                   validationToleranceForCase(backendKind, precision),
                   result);

    result.releaseMs = measureMilliseconds([&]() {
      checkQnnStatus(
          qnnInterface, qnnInterface.contextFree(context, nullptr), "QnnContext_free(updateable_static)");
      context = nullptr;
    });
  } catch (const std::exception& ex) {
    result.status = "ERROR";
    result.detail = ex.what();
    if (context != nullptr && qnnInterface.contextFree != nullptr) qnnInterface.contextFree(context, nullptr);
  }
  return result;
}

std::string renderResults(const AppOptions& options,
                          const std::string& backendPath,
                          BackendKind backendKind,
                          const std::vector<CaseResult>& results) {
  // Render both per-case measurements and a precision-level error summary. The same string is
  // printed to stdout and written to the log file.
  std::ostringstream out;
  std::ostringstream precisions;
  for (size_t i = 0; i < options.precisions.size(); ++i) {
    if (i != 0) precisions << ',';
    precisions << precisionKindName(options.precisions[i]);
  }
  out << "QNN MatMul Static Binary Switch Benchmark\n";
  out << "backend_path : " << backendPath << '\n';
  out << "backend_kind : " << backendKindName(backendKind) << '\n';
  if (backendKind == BackendKind::Htp) {
    out << "htp_soc_model: " << defaultHtpSocModel() << '\n';
    out << "htp_arch     : " << static_cast<int>(defaultHtpArch()) << '\n';
  }
  out << "precisions   : " << precisions.str() << '\n';
  out << "num_weights  : " << options.numWeights << '\n';
  out << "warmup       : " << options.warmup << '\n';
  out << "iterations   : " << options.iterations << '\n';
  out << "log_path     : "
      << (options.logPath.has_value() ? *options.logPath : std::string("(disabled)")) << "\n\n";
  auto appendMetricCell = [](std::ostringstream& stream, int width, double value) {
    if (value < 0.0) {
      stream << std::setw(width) << "-";
    } else {
      stream << std::setw(width) << std::fixed << std::setprecision(4) << value;
    }
  };
  auto appendDashCell = [](std::ostringstream& stream, int width) {
    stream << std::setw(width) << "-";
  };
  out << std::left << std::setw(24) << "shape"
      << std::setw(12) << "precision"
      << std::setw(18) << "mode"
      << std::setw(10) << "status"
      << std::right << std::setw(10) << "valid"
      << std::setw(14) << "prep(ms)"
      << std::setw(14) << "ssd_d(ms)"
      << std::setw(14) << "ssd_g(ms)"
      << std::setw(14) << "htp_w(ms)"
      << std::setw(14) << "final(ms)"
      << std::setw(14) << "first(ms)"
      << std::setw(18) << "exec(ms)"
      << std::setw(14) << "free(ms)"
      << std::setw(15) << "binary(B)"
      << std::setw(15) << "static(B)"
      << std::setw(15) << "runtime(B)"
      << std::setw(14) << "max_err"
      << std::setw(14) << "mean_err" << '\n';
  // ssd_d(ms): tensor file read; ssd_g(ms): graph binary read; htp_w(ms): backend load/update.
  out << std::string(285, '-') << '\n';
  for (const auto& result : results) {
    std::ostringstream shapeText;
    shapeText << result.shape.rows << "x" << result.shape.inner << "x" << result.shape.cols;
    out << std::left << std::setw(24) << shapeText.str()
        << std::setw(12) << precisionKindName(result.precision)
        << std::setw(18) << result.mode
        << std::setw(10) << result.status;
    if (result.status == "OK") {
      out << std::right << std::setw(10) << (result.validationPassed ? "PASS" : "FAIL")
          << std::setw(14) << std::fixed << std::setprecision(4) << result.prepareMs;
      appendMetricCell(out, 14, result.ssdDataLoadMs);
      appendMetricCell(out, 14, result.ssdGraphLoadMs);
      appendMetricCell(out, 14, result.htpWriteMs);
      out << std::setw(14) << result.finalizeMs
          << std::setw(14) << result.firstMs
          << std::setw(18) << formatMeanStd(result.avgMs, result.stdMs)
          << std::setw(14) << result.releaseMs
          << std::setw(15) << result.binaryBytes
          << std::setw(15) << result.staticBytes
          << std::setw(15) << result.runtimeBytes
          << std::setw(14) << result.maxAbsError
          << std::setw(14) << result.meanAbsError << '\n';
    } else {
      out << std::right << std::setw(10) << "-";
      appendDashCell(out, 14);
      appendDashCell(out, 14);
      appendDashCell(out, 14);
      appendDashCell(out, 14);
      appendDashCell(out, 14);
      appendDashCell(out, 14);
      appendDashCell(out, 18);
      appendDashCell(out, 14);
      appendDashCell(out, 15);
      appendDashCell(out, 15);
      appendDashCell(out, 15);
      appendDashCell(out, 14);
      appendDashCell(out, 14);
      out << '\n';
      if (!result.detail.empty()) out << "  detail: " << result.detail << '\n';
    }
  }

  out << "\nprecision_error_summary\n";
  out << std::left << std::setw(12) << "precision"
      << std::right << std::setw(10) << "cases"
      << std::setw(14) << "max_err"
      << std::setw(18) << "avg_mean_err" << '\n';
  out << std::string(54, '-') << '\n';
  for (PrecisionKind precision : options.precisions) {
    uint32_t cases = 0;
    double maxError = 0.0;
    double sumMeanError = 0.0;
    for (const auto& result : results) {
      if (result.precision != precision || result.status != "OK") continue;
      ++cases;
      maxError = std::max(maxError, result.maxAbsError);
      sumMeanError += result.meanAbsError;
    }
    if (cases == 0) {
      out << std::left << std::setw(12) << precisionKindName(precision)
          << std::right << std::setw(10) << 0
          << std::setw(14) << "-"
          << std::setw(18) << "-" << '\n';
      continue;
    }
    out << std::left << std::setw(12) << precisionKindName(precision)
        << std::right << std::setw(10) << cases
        << std::setw(14) << std::fixed << std::setprecision(4) << maxError
        << std::setw(18) << (sumMeanError / static_cast<double>(cases)) << '\n';
  }
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  void* backendLibraryHandle = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device   = nullptr;
  Qnn_LogHandle_t logger      = nullptr;
  const QnnDevice_PlatformInfo_t* platformInfo = nullptr;
  QNN_INTERFACE_VER_TYPE qnnInterface{};
  bool qnnInterfaceReady = false;
  CleanupRegistry cleanupRegistry;

  try {
    const AppOptions options      = parseCommandLine(argc, argv);
    const std::string backendPath = options.backendPath;
    const BackendKind backendKind = detectBackendKind(backendPath);

    backendLibraryHandle = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (backendLibraryHandle == nullptr) {
      throw std::runtime_error("dlopen failed: " + std::string(dlerror()));
    }
    auto getProviders = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
        dlsym(backendLibraryHandle, "QnnInterface_getProviders"));
    if (getProviders == nullptr) throw std::runtime_error("dlsym(QnnInterface_getProviders) failed");

    const QnnInterface_t** providers = nullptr;
    uint32_t numProviders            = 0;
    const auto providerStatus        = getProviders(&providers, &numProviders);
    if (providerStatus != QNN_SUCCESS || providers == nullptr || numProviders == 0) {
      throw std::runtime_error("QnnInterface_getProviders failed");
    }
    for (uint32_t i = 0; i < numProviders; ++i) {
      const auto& apiVersion = providers[i]->apiVersion.coreApiVersion;
      if (apiVersion.major == QNN_API_VERSION_MAJOR && apiVersion.minor >= QNN_API_VERSION_MINOR) {
        qnnInterface      = providers[i]->QNN_INTERFACE_VER_NAME;
        qnnInterfaceReady = true;
        break;
      }
    }
    if (!qnnInterfaceReady) throw std::runtime_error("unable to find a compatible QNN interface provider");

    if (qnnInterface.logCreate != nullptr) {
      const auto logStatus =
          qnnInterface.logCreate(qnnLogCallback, parseLogLevelFromEnv(), &logger);
      if (logStatus == QNN_COMMON_ERROR_NOT_SUPPORTED) {
        logger = nullptr;
      } else {
        checkQnnStatus(qnnInterface, logStatus, "QnnLog_create");
      }
    }

    checkQnnStatus(
        qnnInterface, qnnInterface.backendCreate(logger, nullptr, &backend), "QnnBackend_create");
    createDevice(qnnInterface, logger, backendKind, &device, &platformInfo);

    std::vector<CaseResult> results;
    for (Shape shape : options.shapes) {
      for (PrecisionKind precision : options.precisions) {
        // Precision-specific buffers are rebuilt for each precision because data type and
        // quantization parameters are part of the QNN tensor metadata. The fp32 source and
        // expected-output vectors are scoped here so timed cases do not keep them resident.
        ShapeData data;
        {
          ShapeFloatData floatData = makeShapeFloatData(options, shape);
          data = makeShapeData(options, backendKind, precision, shape, floatData);
        }
        cleanupRegistry.add(data.artifacts.inputPath);
        cleanupRegistry.addAll(data.artifacts.weightPaths);
        cleanupRegistry.addAll(data.expectedOutputPaths);
        clearTensorStorage(data.outputBuffer);
        results.push_back(
            runAppWrite(qnnInterface, backend, device, backendKind, options, precision, shape, data));
        clearTensorStorage(data.outputBuffer);
        results.push_back(runStaticTempLoad(
            qnnInterface, backend, device, backendKind, options, precision, shape, data));
        clearTensorStorage(data.outputBuffer);
        results.push_back(runUpdateableStatic(
            qnnInterface, backend, device, backendKind, options, precision, shape, data));
      }
    }
    for (const auto& result : results) {
      if (result.mode != kModeStaticLoad || result.status != "OK") continue;
      const std::string stem =
          makeArtifactStem(kModeStaticLoad, result.precision, result.shape);
      for (uint32_t i = 0; i < options.numWeights; ++i) {
        cleanupRegistry.add(stem + "_graph_" + std::to_string(i) + ".bin");
      }
    }

    const std::string output = renderResults(options, backendPath, backendKind, results);
    std::cout << output;
    if (options.logPath.has_value()) {
      std::ofstream log(*options.logPath, std::ios::out | std::ios::trunc);
      if (!log) throw std::runtime_error("failed to open log file: " + *options.logPath);
      log << output;
    }

    bool allPassed = true;
    for (const auto& result : results) {
      if (result.status == "SKIP") continue;
      allPassed = allPassed && result.status == "OK" && result.validationPassed;
    }

    if (device != nullptr && qnnInterface.deviceFree != nullptr) {
      checkQnnStatus(qnnInterface, qnnInterface.deviceFree(device), "QnnDevice_free");
      device = nullptr;
    }
    if (platformInfo != nullptr && qnnInterface.deviceFreePlatformInfo != nullptr) {
      checkQnnStatus(qnnInterface,
                     qnnInterface.deviceFreePlatformInfo(logger, platformInfo),
                     "QnnDevice_freePlatformInfo");
      platformInfo = nullptr;
    }
    if (backend != nullptr) {
      checkQnnStatus(qnnInterface, qnnInterface.backendFree(backend), "QnnBackend_free");
      backend = nullptr;
    }
    if (logger != nullptr && qnnInterface.logFree != nullptr) {
      checkQnnStatus(qnnInterface, qnnInterface.logFree(logger), "QnnLog_free");
      logger = nullptr;
    }
    if (backendLibraryHandle != nullptr) {
      dlclose(backendLibraryHandle);
      backendLibraryHandle = nullptr;
    }
    cleanupRegistry.cleanup();
    return allPassed ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << '\n';
    if (qnnInterfaceReady && device != nullptr && qnnInterface.deviceFree != nullptr) {
      qnnInterface.deviceFree(device);
    }
    if (qnnInterfaceReady && platformInfo != nullptr &&
        qnnInterface.deviceFreePlatformInfo != nullptr) {
      qnnInterface.deviceFreePlatformInfo(logger, platformInfo);
    }
    if (qnnInterfaceReady && backend != nullptr && qnnInterface.backendFree != nullptr) {
      qnnInterface.backendFree(backend);
    }
    if (qnnInterfaceReady && logger != nullptr && qnnInterface.logFree != nullptr) {
      qnnInterface.logFree(logger);
    }
    if (backendLibraryHandle != nullptr) dlclose(backendLibraryHandle);
    cleanupRegistry.cleanup();
    return 1;
  }
}
