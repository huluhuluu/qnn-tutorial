#include <QNN/QnnInterface.h>
#include <QNN/QnnLog.h>
#include <QNN/QnnOpDef.h>
#include <QNN/QnnTypes.h>
#include <QNN/HTP/QnnHtpDevice.h>

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
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using QnnInterfaceGetProvidersFn_t =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t*** providerList, uint32_t* numProviders);

enum class BackendKind {
  Cpu,
  Gpu,
  Htp,
  Unknown
};

enum class PrecisionKind {
  Fp32,
  Int8,
  Int16,
  Fp16
};

struct AppOptions {
  std::string backendPath;
  bool dumpBackendInfo = false;
  uint32_t rows        = 256;
  uint32_t inner       = 512;
  uint32_t cols        = 512;
  uint32_t warmup      = 3;
  uint32_t iterations  = 10;
  std::vector<PrecisionKind> precisions = {
      PrecisionKind::Fp32, PrecisionKind::Int8, PrecisionKind::Int16, PrecisionKind::Fp16};
};

struct CapabilitySpec {
  const char* name;
  QnnProperty_Key_t key;
};

struct ValidationResult {
  double maxAbsError  = 0.0;
  double meanAbsError = 0.0;
};

struct BenchmarkResult {
  PrecisionKind precision = PrecisionKind::Int8;
  std::string status      = "UNSUPPORTED";
  std::string detail      = "";
  double initMs           = 0.0;
  double buildMs          = 0.0;
  double avgExecuteMs     = 0.0;
  double stdExecuteMs     = 0.0;
  double releaseMs        = 0.0;
  double totalMs          = 0.0;
  double maxAbsError      = 0.0;
  double meanAbsError     = 0.0;
};

enum class SupportProbeState {
  Supported,
  Unsupported,
  FallbackToDocumentation
};

struct SupportProbeResult {
  SupportProbeState state = SupportProbeState::FallbackToDocumentation;
  std::string detail      = "";
};

struct TensorBuffer {
  Qnn_DataType_t dataType                = QNN_DATATYPE_UNDEFINED;
  Qnn_QuantizeParams_t quantizeParams    = QNN_QUANTIZE_PARAMS_INIT;
  std::vector<Qnn_ScaleOffset_t> axisScaleOffsets = {};
  std::vector<uint8_t> storage           = {};
  size_t elements                        = 0;

  void refreshQuantizationPointers() {
    if (quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      quantizeParams.axisScaleOffsetEncoding.numScaleOffsets =
          static_cast<uint32_t>(axisScaleOffsets.size());
      quantizeParams.axisScaleOffsetEncoding.scaleOffset =
          axisScaleOffsets.empty() ? nullptr : axisScaleOffsets.data();
    }
  }

  void* data() {
    return storage.empty() ? nullptr : storage.data();
  }

  const void* data() const {
    return storage.empty() ? nullptr : storage.data();
  }

  uint32_t sizeInBytes() const {
    return static_cast<uint32_t>(storage.size());
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

std::string valueOrPlaceholder(const char* value) {
  return (value == nullptr || *value == '\0') ? "<null>" : value;
}

bool parseBoolValue(const std::string& value) {
  const std::string lowered = toLower(value);
  return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
}

bool parseBoolEnv(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return false;
  }
  return parseBoolValue(raw);
}

uint32_t parseUintOption(const char* raw, const std::string& flagName) {
  if (raw == nullptr || *raw == '\0') {
    throw std::runtime_error(flagName + " requires a value");
  }
  char* end = nullptr;
  const auto value = std::strtoul(raw, &end, 10);
  if (end == raw || *end != '\0') {
    throw std::runtime_error("invalid value for " + flagName + ": " + raw);
  }
  return static_cast<uint32_t>(value);
}

const char* precisionKindName(PrecisionKind precision) {
  switch (precision) {
    case PrecisionKind::Fp32:
      return "fp32";
    case PrecisionKind::Int8:
      return "int8";
    case PrecisionKind::Int16:
      return "int16";
    case PrecisionKind::Fp16:
      return "fp16";
  }
  return "unknown";
}

PrecisionKind parsePrecisionKind(const std::string& raw) {
  const std::string value = toLower(raw);
  if (value == "fp32") {
    return PrecisionKind::Fp32;
  }
  if (value == "int8") {
    return PrecisionKind::Int8;
  }
  if (value == "int16") {
    return PrecisionKind::Int16;
  }
  if (value == "fp16") {
    return PrecisionKind::Fp16;
  }
  throw std::runtime_error("unknown precision: " + raw);
}

std::vector<PrecisionKind> parsePrecisionList(const std::string& raw) {
  std::vector<PrecisionKind> precisions = {};
  size_t begin                          = 0;
  while (begin < raw.size()) {
    const size_t end = raw.find(',', begin);
    const std::string token =
        raw.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (!token.empty()) {
      precisions.push_back(parsePrecisionKind(token));
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  if (precisions.empty()) {
    throw std::runtime_error("--precisions produced an empty list");
  }
  return precisions;
}

void printUsage(const char* programName) {
  std::cout
      << "Usage: " << programName
      << " [--backend <path>] [--dump-backend-info] [--m <rows>] [--k <inner>] [--n <cols>]\n"
      << "       [--warmup <count>] [--iters <count>] [--precisions fp32,int8,int16,fp16]\n"
      << "       [backend_path]\n"
      << "  --backend <path>       Explicit backend shared library path.\n"
      << "  --dump-backend-info    Print provider/interface/capability info.\n"
      << "  --m <rows>             Left matrix rows. Default: 256.\n"
      << "  --k <inner>            Reduction dimension. Default: 512.\n"
      << "  --n <cols>             Right matrix cols. Default: 512.\n"
      << "  --warmup <count>       Warmup iterations. Default: 3.\n"
      << "  --iters <count>        Timed iterations. Default: 10.\n"
      << "  --precisions <list>    Comma-separated list. Default: fp32,int8,int16,fp16.\n"
      << "  QNN_DUMP_BACKEND_INFO=1 also enables backend info dump.\n";
}

AppOptions parseCommandLine(int argc, char** argv) {
  AppOptions options{};
  options.backendPath      = defaultBackendPath();
  options.dumpBackendInfo  = parseBoolEnv("QNN_DUMP_BACKEND_INFO");
  bool backendPathWasSet   = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    }
    if (arg == "--dump-backend-info") {
      options.dumpBackendInfo = true;
      continue;
    }
    if (arg == "--backend") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--backend requires a value");
      }
      options.backendPath = argv[++i];
      backendPathWasSet   = true;
      continue;
    }
    if (arg == "--m") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--m requires a value");
      }
      options.rows = parseUintOption(argv[++i], "--m");
      continue;
    }
    if (arg == "--k") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--k requires a value");
      }
      options.inner = parseUintOption(argv[++i], "--k");
      continue;
    }
    if (arg == "--n") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--n requires a value");
      }
      options.cols = parseUintOption(argv[++i], "--n");
      continue;
    }
    if (arg == "--warmup") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--warmup requires a value");
      }
      options.warmup = parseUintOption(argv[++i], "--warmup");
      continue;
    }
    if (arg == "--iters") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--iters requires a value");
      }
      options.iterations = parseUintOption(argv[++i], "--iters");
      continue;
    }
    if (arg == "--precisions") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--precisions requires a value");
      }
      options.precisions = parsePrecisionList(argv[++i]);
      continue;
    }
    if (arg.rfind("--", 0) == 0) {
      throw std::runtime_error("unknown option: " + arg);
    }
    if (backendPathWasSet) {
      throw std::runtime_error("multiple backend paths were provided");
    }
    options.backendPath = arg;
    backendPathWasSet   = true;
  }

  if (options.rows == 0 || options.inner == 0 || options.cols == 0) {
    throw std::runtime_error("matrix dimensions must be non-zero");
  }
  if (options.iterations == 0) {
    throw std::runtime_error("--iters must be non-zero");
  }
  return options;
}

BackendKind detectBackendKind(const std::string& backendPath) {
  const std::string path = toLower(backendPath);
  if (path.find("qnnhtp") != std::string::npos) {
    return BackendKind::Htp;
  }
  if (path.find("qnngpu") != std::string::npos) {
    return BackendKind::Gpu;
  }
  if (path.find("qnncpu") != std::string::npos) {
    return BackendKind::Cpu;
  }
  return BackendKind::Unknown;
}

const char* backendKindName(BackendKind kind) {
  switch (kind) {
    case BackendKind::Cpu:
      return "CPU";
    case BackendKind::Gpu:
      return "GPU";
    case BackendKind::Htp:
      return "NPU";
    case BackendKind::Unknown:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

const char* backendIdName(uint32_t backendId) {
  switch (backendId) {
    case QNN_BACKEND_ID_NULL:
      return "NULL";
    case 1:
      return "REFERENCE";
    case 2:
      return "SAVER";
    case 3:
      return "CPU";
    case 4:
      return "GPU";
    case 6:
      return "HTP";
  }
  return "UNKNOWN";
}

std::string versionToString(const Qnn_Version_t& version) {
  return std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
         std::to_string(version.patch);
}

const char* propertyStatusName(Qnn_ErrorHandle_t status) {
  switch (status) {
    case QNN_PROPERTY_SUPPORTED:
      return "supported";
    case QNN_PROPERTY_NOT_SUPPORTED:
      return "not_supported";
    case QNN_PROPERTY_ERROR_UNKNOWN_KEY:
      return "unknown_key";
  }
  return "error";
}

const char* logLevelName(QnnLog_Level_t level) {
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      return "ERROR";
    case QNN_LOG_LEVEL_WARN:
      return "WARN";
    case QNN_LOG_LEVEL_INFO:
      return "INFO";
    case QNN_LOG_LEVEL_VERBOSE:
      return "VERBOSE";
    case QNN_LOG_LEVEL_DEBUG:
      return "DEBUG";
    case QNN_LOG_LEVEL_MAX:
      return "MAX";
  }
  return "UNKNOWN";
}

QnnLog_Level_t parseLogLevelFromEnv() {
  const char* raw = std::getenv("QNN_LOG_LEVEL");
  if (raw == nullptr || *raw == '\0') {
    return QNN_LOG_LEVEL_ERROR;
  }
  const std::string value = toLower(raw);
  if (value == "error") {
    return QNN_LOG_LEVEL_ERROR;
  }
  if (value == "warn") {
    return QNN_LOG_LEVEL_WARN;
  }
  if (value == "info") {
    return QNN_LOG_LEVEL_INFO;
  }
  if (value == "verbose") {
    return QNN_LOG_LEVEL_VERBOSE;
  }
  if (value == "debug") {
    return QNN_LOG_LEVEL_DEBUG;
  }
  return QNN_LOG_LEVEL_ERROR;
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

uint32_t defaultHtpSocModel() {
  return QNN_SOC_MODEL_SM8750;
}

QnnHtpDevice_Arch_t defaultHtpArch() {
  return QNN_HTP_DEVICE_ARCH_V79;
}

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
  if (status != QNN_SUCCESS) {
    throw std::runtime_error(stage + " failed: " + getErrorText(qnnInterface, status));
  }
}

long long measureMicroseconds(const std::function<void()>& fn) {
  const auto start = Clock::now();
  fn();
  const auto stop = Clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
}

double elapsedMilliseconds(Clock::time_point start, Clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

double computeSampleStddev(const std::vector<double>& samples, double mean) {
  if (samples.size() <= 1) {
    return 0.0;
  }

  double sumSquaredDiff = 0.0;
  for (double sample : samples) {
    const double diff = sample - mean;
    sumSquaredDiff += diff * diff;
  }
  return std::sqrt(sumSquaredDiff / static_cast<double>(samples.size() - 1));
}

std::string formatMeanStdMilliseconds(double mean, double stddev) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << mean << "\u00B1" << stddev;
  return stream.str();
}

uint32_t bytesPerElement(Qnn_DataType_t dataType) {
  switch (dataType) {
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return 4;
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_INT_8:
      return 1;
    default:
      throw std::runtime_error("unsupported data type for bytesPerElement");
  }
}

// This is a doc-driven support matrix for MatMul. We use it to pre-mark combinations as
// unsupported before trying to build the graph. It is not a runtime probe.
bool isPrecisionDocumentedSupported(BackendKind backendKind, PrecisionKind precision) {
  switch (backendKind) {
    case BackendKind::Cpu:
      return precision == PrecisionKind::Fp32 || precision == PrecisionKind::Int8;
    case BackendKind::Gpu:
      return precision == PrecisionKind::Fp32 || precision == PrecisionKind::Fp16;
    case BackendKind::Htp:
      return true;
    case BackendKind::Unknown:
      return true;
  }
  return false;
}

std::vector<float> makeInputData(size_t size, int modulo, float scale) {
  std::vector<float> values(size);
  for (size_t i = 0; i < size; ++i) {
    values[i] = static_cast<float>(static_cast<int>(i % modulo) - (modulo / 2)) / scale;
  }
  return values;
}

std::vector<float> referenceMatmul(const std::vector<float>& input,
                                   const std::vector<float>& weights,
                                   uint32_t rows,
                                   uint32_t inner,
                                   uint32_t cols) {
  std::vector<float> output(rows * cols, 0.0f);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < cols; ++col) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < inner; ++k) {
        acc += input[row * inner + k] * weights[k * cols + col];
      }
      output[row * cols + col] = acc;
    }
  }
  return output;
}

uint16_t floatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const uint32_t sign = (bits >> 16) & 0x8000u;
  uint32_t mantissa   = bits & 0x007FFFFFu;
  int32_t exponent    = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa = (mantissa | 0x00800000u) >> static_cast<uint32_t>(1 - exponent);
    if ((mantissa & 0x00001000u) != 0u) {
      mantissa += 0x00002000u;
    }
    return static_cast<uint16_t>(sign | (mantissa >> 13));
  }

  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }

  if ((mantissa & 0x00001000u) != 0u) {
    mantissa += 0x00002000u;
    if ((mantissa & 0x00800000u) != 0u) {
      mantissa = 0;
      ++exponent;
      if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
      }
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

template <typename T>
void writeTypedStorage(TensorBuffer& buffer, const std::vector<T>& values) {
  buffer.elements = values.size();
  buffer.storage.resize(values.size() * sizeof(T));
  if (!values.empty()) {
    std::memcpy(buffer.storage.data(), values.data(), buffer.storage.size());
  }
}

TensorBuffer makeFp16Buffer(const std::vector<float>& values) {
  TensorBuffer buffer{};
  buffer.dataType = QNN_DATATYPE_FLOAT_16;
  std::vector<uint16_t> bits(values.size(), 0u);
  for (size_t i = 0; i < values.size(); ++i) {
    bits[i] = floatToHalfBits(values[i]);
  }
  writeTypedStorage(buffer, bits);
  return buffer;
}

TensorBuffer makeFloat32Buffer(const std::vector<float>& values) {
  TensorBuffer buffer{};
  buffer.dataType = QNN_DATATYPE_FLOAT_32;
  writeTypedStorage(buffer, values);
  return buffer;
}

TensorBuffer makeSignedSymmetricBuffer(const std::vector<float>& values,
                                       Qnn_DataType_t dataType,
                                       int bits) {
  TensorBuffer buffer{};
  buffer.dataType                         = dataType;
  buffer.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  buffer.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;

  float maxAbs = 0.0f;
  for (float value : values) {
    maxAbs = std::max(maxAbs, std::abs(value));
  }
  const int64_t qmax = (static_cast<int64_t>(1) << (bits - 1)) - 1;
  const double scale = (maxAbs <= std::numeric_limits<float>::epsilon()) ? 1.0
                                                                         : (maxAbs / qmax);
  buffer.quantizeParams.scaleOffsetEncoding.scale  = static_cast<float>(scale);
  buffer.quantizeParams.scaleOffsetEncoding.offset = 0;

  if (bits == 8) {
    std::vector<int8_t> quantized(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto q = static_cast<int64_t>(std::llround(values[i] / scale));
      const auto clamped = std::clamp<int64_t>(q, -qmax, qmax);
      quantized[i]       = static_cast<int8_t>(clamped);
    }
    writeTypedStorage(buffer, quantized);
    return buffer;
  }

  if (bits == 16) {
    std::vector<int16_t> quantized(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto q = static_cast<int64_t>(std::llround(values[i] / scale));
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
  buffer.dataType                         = dataType;
  buffer.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  buffer.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;

  const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
  const double minValue = (minIt == values.end()) ? 0.0 : *minIt;
  const double maxValue = (maxIt == values.end()) ? 0.0 : *maxIt;
  const double qmin     = 0.0;
  const double qmax     = static_cast<double>((static_cast<uint64_t>(1) << bits) - 1);
  double scale          = (maxValue - minValue) / std::max(1.0, qmax - qmin);
  if (scale <= 1e-9) {
    scale = 1.0;
  }
  const double zeroPoint = std::clamp(std::llround(-minValue / scale), 0ll, static_cast<long long>(qmax));

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

TensorBuffer makeSignedInt8AxisWeightBuffer(const std::vector<float>& values,
                                            uint32_t rows,
                                            uint32_t cols) {
  TensorBuffer buffer{};
  buffer.dataType                           = QNN_DATATYPE_SFIXED_POINT_8;
  buffer.quantizeParams.encodingDefinition  = QNN_DEFINITION_DEFINED;
  buffer.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  buffer.axisScaleOffsets.resize(cols, QNN_SCALE_OFFSET_INIT);
  buffer.quantizeParams.axisScaleOffsetEncoding.axis = 1;

  std::vector<int8_t> quantized(values.size(), 0);
  constexpr int64_t kQmax = 127;
  for (uint32_t col = 0; col < cols; ++col) {
    float maxAbs = 0.0f;
    for (uint32_t row = 0; row < rows; ++row) {
      maxAbs = std::max(maxAbs, std::abs(values[row * cols + col]));
    }
    const double scale = (maxAbs <= std::numeric_limits<float>::epsilon()) ? 1.0
                                                                           : (maxAbs / kQmax);
    buffer.axisScaleOffsets[col].scale  = static_cast<float>(scale);
    buffer.axisScaleOffsets[col].offset = 0;
    for (uint32_t row = 0; row < rows; ++row) {
      const float value   = values[row * cols + col];
      const auto q        = static_cast<int64_t>(std::llround(value / scale));
      const auto clamped  = std::clamp<int64_t>(q, -kQmax, kQmax);
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
      for (size_t i = 0; i < buffer.elements; ++i) {
        decoded[i] = halfBitsToFloat(ptr[i]);
      }
      return decoded;
    }
    case QNN_DATATYPE_SFIXED_POINT_8: {
      if (buffer.quantizeParams.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        throw std::runtime_error("decode only supports SCALE_OFFSET for output tensor");
      }
      const auto* ptr = reinterpret_cast<const int8_t*>(buffer.data());
      const float scale = buffer.quantizeParams.scaleOffsetEncoding.scale;
      const int32_t offset = buffer.quantizeParams.scaleOffsetEncoding.offset;
      for (size_t i = 0; i < buffer.elements; ++i) {
        decoded[i] = (static_cast<int32_t>(ptr[i]) + offset) * scale;
      }
      return decoded;
    }
    case QNN_DATATYPE_UFIXED_POINT_8: {
      if (buffer.quantizeParams.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        throw std::runtime_error("decode only supports SCALE_OFFSET for output tensor");
      }
      const auto* ptr = reinterpret_cast<const uint8_t*>(buffer.data());
      const float scale = buffer.quantizeParams.scaleOffsetEncoding.scale;
      const int32_t offset = buffer.quantizeParams.scaleOffsetEncoding.offset;
      for (size_t i = 0; i < buffer.elements; ++i) {
        decoded[i] = (static_cast<int32_t>(ptr[i]) + offset) * scale;
      }
      return decoded;
    }
    case QNN_DATATYPE_SFIXED_POINT_16: {
      if (buffer.quantizeParams.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        throw std::runtime_error("decode only supports SCALE_OFFSET for output tensor");
      }
      const auto* ptr = reinterpret_cast<const int16_t*>(buffer.data());
      const float scale = buffer.quantizeParams.scaleOffsetEncoding.scale;
      const int32_t offset = buffer.quantizeParams.scaleOffsetEncoding.offset;
      for (size_t i = 0; i < buffer.elements; ++i) {
        decoded[i] = (static_cast<int32_t>(ptr[i]) + offset) * scale;
      }
      return decoded;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      if (buffer.quantizeParams.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        throw std::runtime_error("decode only supports SCALE_OFFSET for output tensor");
      }
      const auto* ptr = reinterpret_cast<const uint16_t*>(buffer.data());
      const float scale = buffer.quantizeParams.scaleOffsetEncoding.scale;
      const int32_t offset = buffer.quantizeParams.scaleOffsetEncoding.offset;
      for (size_t i = 0; i < buffer.elements; ++i) {
        decoded[i] = (static_cast<int32_t>(ptr[i]) + offset) * scale;
      }
      return decoded;
    }
    default:
      throw std::runtime_error("unsupported output data type for decode");
  }
}

ValidationResult validateOutput(const std::vector<float>& actual,
                                const std::vector<float>& expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("output size mismatch");
  }

  double maxAbsError = 0.0;
  double sumAbsError = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double absError =
        std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
    maxAbsError = std::max(maxAbsError, absError);
    sumAbsError += absError;
  }
  return {maxAbsError, sumAbsError / static_cast<double>(actual.size())};
}

Qnn_Tensor_t makeTensor(const char* name,
                        Qnn_TensorType_t type,
                        Qnn_DataType_t dataType,
                        const Qnn_QuantizeParams_t& quantizeParams,
                        uint32_t* dimensions,
                        uint32_t rank,
                        void* data,
                        uint32_t dataSize) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.version      = QNN_TENSOR_VERSION_1;
  tensor.v1.name      = name;
  tensor.v1.type      = type;
  tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor.v1.dataType   = dataType;
  tensor.v1.quantizeParams = quantizeParams;
  tensor.v1.rank       = rank;
  tensor.v1.dimensions = dimensions;
  tensor.v1.memType    = QNN_TENSORMEMTYPE_RAW;
  tensor.v1.clientBuf.data = data;
  tensor.v1.clientBuf.dataSize = dataSize;
  return tensor;
}

Qnn_Tensor_t makeExecuteTensor(const Qnn_Tensor_t& registeredTensor, void* data, uint32_t dataSize) {
  Qnn_Tensor_t tensor = registeredTensor;
  tensor.v1.clientBuf.data = data;
  tensor.v1.clientBuf.dataSize = dataSize;
  return tensor;
}

SupportProbeResult validateMatmulSupport(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                                         Qnn_BackendHandle_t backend,
                                         BackendKind backendKind,
                                         PrecisionKind precision,
                                         uint32_t rows,
                                         uint32_t inner,
                                         uint32_t cols,
                                         TensorBuffer& inputBuffer,
                                         TensorBuffer& weightBuffer,
                                         TensorBuffer& outputBuffer) {
#if defined(__ANDROID__)
  // On Android targets, HTP validateOpConfig for quantized MatMul can crash inside
  // libQnnHtpPrepare.so even though graph composition/execution succeeds afterwards.
  // Skip that probe and let the real graph build path decide support.
  if (backendKind == BackendKind::Htp && precision == PrecisionKind::Int8) {
    return {SupportProbeState::FallbackToDocumentation,
            "skip runtime probe for Android HTP int8 MatMul"};
  }
#endif

  if (qnnInterface.backendValidateOpConfig == nullptr) {
    return {SupportProbeState::FallbackToDocumentation,
            "runtime probe unavailable: backendValidateOpConfig is null"};
  }

  constexpr uint32_t kTensorRank = 2;
  std::array<uint32_t, kTensorRank> inputDims  = {rows, inner};
  std::array<uint32_t, kTensorRank> weightDims = {inner, cols};
  std::array<uint32_t, kTensorRank> outputDims = {rows, cols};

  const std::string inputName = "probe_input_" + std::string(precisionKindName(precision));
  const std::string weightName = "probe_weight_" + std::string(precisionKindName(precision));
  const std::string outputName = "probe_output_" + std::string(precisionKindName(precision));

  Qnn_Tensor_t inputTensor = makeTensor(inputName.c_str(),
                                        QNN_TENSOR_TYPE_APP_WRITE,
                                        inputBuffer.dataType,
                                        inputBuffer.quantizeParams,
                                        inputDims.data(),
                                        kTensorRank,
                                        nullptr,
                                        0);
  Qnn_Tensor_t weightTensor = makeTensor(weightName.c_str(),
                                         QNN_TENSOR_TYPE_STATIC,
                                         weightBuffer.dataType,
                                         weightBuffer.quantizeParams,
                                         weightDims.data(),
                                         kTensorRank,
                                         weightBuffer.data(),
                                         weightBuffer.sizeInBytes());
  Qnn_Tensor_t outputTensor = makeTensor(outputName.c_str(),
                                         QNN_TENSOR_TYPE_APP_READ,
                                         outputBuffer.dataType,
                                         outputBuffer.quantizeParams,
                                         outputDims.data(),
                                         kTensorRank,
                                         nullptr,
                                         0);

  std::array<Qnn_Tensor_t, 2> nodeInputs = {inputTensor, weightTensor};
  std::array<Qnn_Tensor_t, 1> nodeOutputs = {outputTensor};

  Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
  opConfig.v1.name          = "matmul_probe";
  opConfig.v1.packageName   = QNN_OP_PACKAGE_NAME_QTI_AISW;
  opConfig.v1.typeName      = QNN_OP_MAT_MUL;
  opConfig.v1.numOfParams   = 0;
  opConfig.v1.params        = nullptr;
  opConfig.v1.numOfInputs   = static_cast<uint32_t>(nodeInputs.size());
  opConfig.v1.inputTensors  = nodeInputs.data();
  opConfig.v1.numOfOutputs  = static_cast<uint32_t>(nodeOutputs.size());
  opConfig.v1.outputTensors = nodeOutputs.data();

  const auto status = qnnInterface.backendValidateOpConfig(backend, opConfig);
  switch (status) {
    case QNN_SUCCESS:
      return {SupportProbeState::Supported, ""};
    case QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE:
      return {SupportProbeState::Unsupported,
              "runtime validation rejected this MatMul op config"};
    case QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND:
      return {SupportProbeState::Unsupported,
              "runtime validation could not find a matching op package"};
    case QNN_BACKEND_ERROR_NOT_SUPPORTED:
      return {SupportProbeState::FallbackToDocumentation,
              "runtime probe unavailable: backendValidateOpConfig is not supported"};
    default:
      return {SupportProbeState::FallbackToDocumentation,
              "runtime probe failed: " + getErrorText(qnnInterface, status)};
  }
}

void printApiAvailability(const QNN_INTERFACE_VER_TYPE& qnnInterface) {
  std::cout << "api_entries\n";
  const std::array<std::pair<const char*, bool>, 10> apiEntries = {{
      {"propertyHasCapability", qnnInterface.propertyHasCapability != nullptr},
      {"backendGetApiVersion", qnnInterface.backendGetApiVersion != nullptr},
      {"backendGetBuildId", qnnInterface.backendGetBuildId != nullptr},
      {"deviceCreate", qnnInterface.deviceCreate != nullptr},
      {"deviceGetPlatformInfo", qnnInterface.deviceGetPlatformInfo != nullptr},
      {"contextCreate", qnnInterface.contextCreate != nullptr},
      {"graphCreate", qnnInterface.graphCreate != nullptr},
      {"tensorCreateGraphTensor", qnnInterface.tensorCreateGraphTensor != nullptr},
      {"graphFinalize", qnnInterface.graphFinalize != nullptr},
      {"graphExecute", qnnInterface.graphExecute != nullptr},
  }};
  for (const auto& [name, available] : apiEntries) {
    std::cout << "  " << std::left << std::setw(34) << name << (available ? "yes" : "no")
              << '\n';
  }
}

void printCapabilitySummary(const QNN_INTERFACE_VER_TYPE& qnnInterface) {
  std::cout << "capabilities\n";
  if (qnnInterface.propertyHasCapability == nullptr) {
    std::cout << "  " << std::left << std::setw(34) << "propertyHasCapability" << "unavailable\n";
    return;
  }

  const std::array<CapabilitySpec, 9> capabilities = {{
      {"device_api_group", QNN_PROPERTY_GROUP_DEVICE},
      {"backend_support_op_package", QNN_PROPERTY_BACKEND_SUPPORT_OP_PACKAGE},
      {"backend_support_composition", QNN_PROPERTY_BACKEND_SUPPORT_COMPOSITION},
      {"context_support_caching", QNN_PROPERTY_CONTEXT_SUPPORT_CACHING},
      {"graph_support_execute", QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE},
      {"graph_support_async_execution", QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION},
      {"graph_support_online_prepare", QNN_PROPERTY_GRAPH_SUPPORT_ONLINE_PREPARE},
      {"tensor_support_context_tensors", QNN_PROPERTY_TENSOR_SUPPORT_CONTEXT_TENSORS},
      {"tensor_support_dynamic_dimensions", QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS},
  }};

  for (const auto& capability : capabilities) {
    const auto status = qnnInterface.propertyHasCapability(capability.key);
    std::cout << "  " << std::left << std::setw(34) << capability.name
              << propertyStatusName(status) << '\n';
  }
}

void dumpBackendInfo(const std::string& backendPath,
                     BackendKind backendKind,
                     const QnnInterface_t* const* providers,
                     uint32_t numProviders,
                     const QnnInterface_t* selectedProvider,
                     const QNN_INTERFACE_VER_TYPE& qnnInterface) {
  std::cout << "\nBackend Info\n";
  std::cout << "backend_path  : " << backendPath << '\n';
  std::cout << "backend_kind  : " << backendKindName(backendKind) << '\n';
  std::cout << "provider_count: " << numProviders << '\n';

  for (uint32_t i = 0; i < numProviders; ++i) {
    const QnnInterface_t* provider = providers[i];
    if (provider == nullptr) {
      continue;
    }
    std::cout << "provider[" << i << "]\n";
    std::cout << "  backend_id        : " << provider->backendId << " ("
              << backendIdName(provider->backendId) << ")\n";
    std::cout << "  provider_name     : " << valueOrPlaceholder(provider->providerName) << '\n';
    std::cout << "  core_api_version  : "
              << versionToString(provider->apiVersion.coreApiVersion) << '\n';
    std::cout << "  backend_api_ver   : "
              << versionToString(provider->apiVersion.backendApiVersion) << '\n';
    std::cout << "  selected          : " << (provider == selectedProvider ? "yes" : "no")
              << '\n';
  }

  if (qnnInterface.backendGetApiVersion != nullptr) {
    Qnn_ApiVersion_t backendApiVersion = QNN_API_VERSION_INIT;
    const auto status                  = qnnInterface.backendGetApiVersion(&backendApiVersion);
    std::cout << "selected_core_api   : ";
    if (status == QNN_SUCCESS) {
      std::cout << versionToString(backendApiVersion.coreApiVersion) << '\n';
      std::cout << "selected_backend_api: "
                << versionToString(backendApiVersion.backendApiVersion) << '\n';
    } else {
      std::cout << "unavailable (" << getErrorText(qnnInterface, status) << ")\n";
    }
  } else {
    std::cout << "selected_core_api   : unavailable\n";
  }

  if (qnnInterface.backendGetBuildId != nullptr) {
    const char* buildId  = nullptr;
    const auto status    = qnnInterface.backendGetBuildId(&buildId);
    std::cout << "backend_build_id: ";
    if (status == QNN_SUCCESS) {
      std::cout << valueOrPlaceholder(buildId) << '\n';
    } else {
      std::cout << "unavailable (" << getErrorText(qnnInterface, status) << ")\n";
    }
  } else {
    std::cout << "backend_build_id: unavailable\n";
  }

  printApiAvailability(qnnInterface);
  printCapabilitySummary(qnnInterface);
}

void createDeviceWithFallback(const QNN_INTERFACE_VER_TYPE& qnnInterface,
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
    if (status == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      *device = nullptr;
    }
    return;
  }

  if (backendKind == BackendKind::Htp) {
    if (qnnInterface.deviceGetPlatformInfo != nullptr) {
      const auto infoStatus = qnnInterface.deviceGetPlatformInfo(logger, platformInfo);
      if (infoStatus == QNN_SUCCESS && *platformInfo != nullptr) {
        QnnDevice_Config_t platformConfig = QNN_DEVICE_CONFIG_INIT;
        platformConfig.option       = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
        platformConfig.hardwareInfo = const_cast<QnnDevice_PlatformInfo_t*>(*platformInfo);
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
  if (status == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
    *device = nullptr;
  }
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
        outputBuffer = makeUnsignedAffineBuffer(expectedOutputFloat, QNN_DATATYPE_UFIXED_POINT_8, 8);
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

BenchmarkResult runBenchmarkCase(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                                 Qnn_BackendHandle_t backend,
                                 Qnn_DeviceHandle_t device,
                                 BackendKind backendKind,
                                 PrecisionKind precision,
                                 uint32_t rows,
                                 uint32_t inner,
                                 uint32_t cols,
                                 uint32_t warmup,
                                 uint32_t iterations) {
  BenchmarkResult result{};
  result.precision = precision;

  Qnn_ContextHandle_t context = nullptr;
  const auto caseStart        = Clock::now();
  try {
    const auto initStart = Clock::now();
    const std::vector<float> inputDataFloat  = makeInputData(rows * inner, 37, 19.0f);
    const std::vector<float> weightDataFloat = makeInputData(inner * cols, 29, 17.0f);
    const std::vector<float> expectedOutputFloat =
        referenceMatmul(inputDataFloat, weightDataFloat, rows, inner, cols);

    TensorBuffer inputBuffer{};
    TensorBuffer weightBuffer{};
    TensorBuffer outputBuffer{};
    prepareBuffers(backendKind,
                   precision,
                   inner,
                   cols,
                   inputDataFloat,
                   weightDataFloat,
                   expectedOutputFloat,
                   inputBuffer,
                   weightBuffer,
                   outputBuffer);

    inputBuffer.refreshQuantizationPointers();
    weightBuffer.refreshQuantizationPointers();
    outputBuffer.refreshQuantizationPointers();

    const SupportProbeResult supportProbe = validateMatmulSupport(qnnInterface,
                                                                  backend,
                                                                  backendKind,
                                                                  precision,
                                                                  rows,
                                                                  inner,
                                                                  cols,
                                                                  inputBuffer,
                                                                  weightBuffer,
                                                                  outputBuffer);
    if (supportProbe.state == SupportProbeState::Unsupported) {
      result.status = "UNSUPPORTED";
      result.detail = supportProbe.detail;
      return result;
    }

    if (supportProbe.state == SupportProbeState::FallbackToDocumentation &&
        !isPrecisionDocumentedSupported(backendKind, precision)) {
      result.status = "UNSUPPORTED";
      result.detail = supportProbe.detail.empty()
                          ? "documented as unsupported by this backend"
                          : supportProbe.detail + "; documented as unsupported by this backend";
      return result;
    }

    result.status = "OK";

    checkQnnStatus(qnnInterface,
                   qnnInterface.contextCreate(backend, device, nullptr, &context),
                   "QnnContext_create");

    Qnn_GraphHandle_t graph = nullptr;
    checkQnnStatus(
        qnnInterface, qnnInterface.graphCreate(context, "precision_bench", nullptr, &graph), "QnnGraph_create");
    const auto initStop = Clock::now();
    result.initMs       = elapsedMilliseconds(initStart, initStop);

    constexpr uint32_t kTensorRank = 2;
    std::array<uint32_t, kTensorRank> inputDims  = {rows, inner};
    std::array<uint32_t, kTensorRank> weightDims = {inner, cols};
    std::array<uint32_t, kTensorRank> outputDims = {rows, cols};
    const std::string inputName   = "input_" + std::string(precisionKindName(precision));
    const std::string weightName  = "weight_" + std::string(precisionKindName(precision));
    const std::string outputName  = "output_" + std::string(precisionKindName(precision));

    const auto buildStart = Clock::now();
    Qnn_Tensor_t inputTensor = makeTensor(inputName.c_str(),
                                          QNN_TENSOR_TYPE_APP_WRITE,
                                          inputBuffer.dataType,
                                          inputBuffer.quantizeParams,
                                          inputDims.data(),
                                          kTensorRank,
                                          nullptr,
                                          0);
    Qnn_Tensor_t weightTensor = makeTensor(weightName.c_str(),
                                           QNN_TENSOR_TYPE_STATIC,
                                           weightBuffer.dataType,
                                           weightBuffer.quantizeParams,
                                           weightDims.data(),
                                           kTensorRank,
                                           weightBuffer.data(),
                                           weightBuffer.sizeInBytes());
    Qnn_Tensor_t outputTensor = makeTensor(outputName.c_str(),
                                           QNN_TENSOR_TYPE_APP_READ,
                                           outputBuffer.dataType,
                                           outputBuffer.quantizeParams,
                                           outputDims.data(),
                                           kTensorRank,
                                           nullptr,
                                           0);

    checkQnnStatus(qnnInterface,
                   qnnInterface.tensorCreateGraphTensor(graph, &inputTensor),
                   "QnnTensor_createGraphTensor(input)");
    checkQnnStatus(qnnInterface,
                   qnnInterface.tensorCreateGraphTensor(graph, &weightTensor),
                   "QnnTensor_createGraphTensor(weight)");
    checkQnnStatus(qnnInterface,
                   qnnInterface.tensorCreateGraphTensor(graph, &outputTensor),
                   "QnnTensor_createGraphTensor(output)");

    std::array<Qnn_Tensor_t, 2> nodeInputs = {inputTensor, weightTensor};
    std::array<Qnn_Tensor_t, 1> nodeOutputs = {outputTensor};

    Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
    opConfig.v1.name          = "matmul_bench";
    opConfig.v1.packageName   = QNN_OP_PACKAGE_NAME_QTI_AISW;
    opConfig.v1.typeName      = QNN_OP_MAT_MUL;
    opConfig.v1.numOfParams   = 0;
    opConfig.v1.params        = nullptr;
    opConfig.v1.numOfInputs   = static_cast<uint32_t>(nodeInputs.size());
    opConfig.v1.inputTensors  = nodeInputs.data();
    opConfig.v1.numOfOutputs  = static_cast<uint32_t>(nodeOutputs.size());
    opConfig.v1.outputTensors = nodeOutputs.data();

    checkQnnStatus(qnnInterface, qnnInterface.graphAddNode(graph, opConfig), "QnnGraph_addNode");

    checkQnnStatus(
        qnnInterface, qnnInterface.graphFinalize(graph, nullptr, nullptr), "QnnGraph_finalize");
    const auto buildStop = Clock::now();
    result.buildMs       = elapsedMilliseconds(buildStart, buildStop);

    Qnn_Tensor_t executeInput =
        makeExecuteTensor(inputTensor, inputBuffer.data(), inputBuffer.sizeInBytes());
    Qnn_Tensor_t executeOutput =
        makeExecuteTensor(outputTensor, outputBuffer.data(), outputBuffer.sizeInBytes());
    Qnn_Tensor_t executeInputs[]  = {executeInput};
    Qnn_Tensor_t executeOutputs[] = {executeOutput};

    auto executeGraph = [&]() {
      checkQnnStatus(qnnInterface,
                     qnnInterface.graphExecute(
                         graph, executeInputs, 1, executeOutputs, 1, nullptr, nullptr),
                     "QnnGraph_execute");
    };

    for (uint32_t i = 0; i < warmup; ++i) {
      executeGraph();
    }

    std::vector<double> samplesMs(iterations, 0.0);
    for (uint32_t i = 0; i < iterations; ++i) {
      const auto sampleUs = measureMicroseconds(executeGraph);
      samplesMs[i]        = static_cast<double>(sampleUs) / 1000.0;
    }

    result.avgExecuteMs =
        std::accumulate(samplesMs.begin(), samplesMs.end(), 0.0) / static_cast<double>(samplesMs.size());
    result.stdExecuteMs = computeSampleStddev(samplesMs, result.avgExecuteMs);

    const std::vector<float> actualOutput = decodeTensorToFloat(outputBuffer);
    const ValidationResult validation = validateOutput(actualOutput, expectedOutputFloat);
    result.maxAbsError  = validation.maxAbsError;
    result.meanAbsError = validation.meanAbsError;

    const auto releaseStart = Clock::now();
    checkQnnStatus(qnnInterface, qnnInterface.contextFree(context, nullptr), "QnnContext_free");
    context = nullptr;
    const auto releaseStop = Clock::now();
    result.releaseMs       = elapsedMilliseconds(releaseStart, releaseStop);
    result.totalMs         = elapsedMilliseconds(caseStart, releaseStop);
  } catch (const std::exception& ex) {
    result.status = "ERROR";
    result.detail = ex.what();
    const auto releaseStart = Clock::now();
    if (context != nullptr && qnnInterface.contextFree != nullptr) {
      qnnInterface.contextFree(context, nullptr);
      context = nullptr;
    }
    const auto releaseStop = Clock::now();
    if (context == nullptr) {
      result.releaseMs = elapsedMilliseconds(releaseStart, releaseStop);
    }
    result.totalMs = elapsedMilliseconds(caseStart, releaseStop);
  }
  return result;
}

void printBenchmarkResults(BackendKind backendKind,
                           const AppOptions& options,
                           const std::vector<BenchmarkResult>& results) {
  std::cout << "\nQNN MatMul Precision Benchmark\n";
  std::cout << "backend_kind : " << backendKindName(backendKind) << '\n';
  std::cout << "backend_path : " << options.backendPath << '\n';
  if (backendKind == BackendKind::Htp) {
    std::cout << "htp_soc_model: " << defaultHtpSocModel() << '\n';
    std::cout << "htp_arch     : " << static_cast<int>(defaultHtpArch()) << '\n';
  }
  std::cout << "shape        : A[" << options.rows << ", " << options.inner << "] x B[" << options.inner
            << ", " << options.cols << "] -> C[" << options.rows << ", " << options.cols << "]\n";
  std::cout << "warmup       : " << options.warmup << '\n';
  std::cout << "iterations   : " << options.iterations << '\n';

  std::cout << "\nResults\n";
  std::cout << std::left << std::setw(10) << "precision" << std::setw(14) << "status"
            << std::right << std::setw(14) << "init(ms)" << std::setw(14) << "build(ms)"
            << std::setw(22) << "exec(ms)"
            << std::setw(16) << "release(ms)" << std::setw(14) << "total(ms)"
            << std::setw(16) << "max_abs_err" << std::setw(16) << "mean_abs_err" << '\n';
  std::cout << std::string(136, '-') << '\n';

  std::cout << std::fixed << std::setprecision(4);
  for (const auto& result : results) {
    std::cout << std::left << std::setw(10) << precisionKindName(result.precision)
              << std::setw(14) << result.status;
    if (result.status == "OK") {
      const std::string execTime = formatMeanStdMilliseconds(result.avgExecuteMs, result.stdExecuteMs);
      std::cout << std::right << std::setw(14) << result.initMs << std::setw(14)
                << result.buildMs << std::setw(22) << execTime << std::setw(16) << result.releaseMs << std::setw(14)
                << result.totalMs << std::setw(16)
                << result.maxAbsError << std::setw(16) << result.meanAbsError << '\n';
    } else {
      std::cout << std::right << std::setw(14) << "-" << std::setw(14) << "-" << std::setw(22)
                << "-" << std::setw(16) << "-" << std::setw(14) << "-"
                << std::setw(16) << "-" << std::setw(16) << "-" << '\n';
      if (!result.detail.empty()) {
        std::cout << "  detail: " << result.detail << '\n';
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const AppOptions options = parseCommandLine(argc, argv);
  const BackendKind backendKind = detectBackendKind(options.backendPath);
  const QnnLog_Level_t logLevel = parseLogLevelFromEnv();

  void* backendLibraryHandle             = nullptr;
  Qnn_BackendHandle_t backend            = nullptr;
  Qnn_DeviceHandle_t device              = nullptr;
  Qnn_LogHandle_t logger                 = nullptr;
  const QnnDevice_PlatformInfo_t* platformInfo = nullptr;
  QNN_INTERFACE_VER_TYPE qnnInterface{};
  bool qnnInterfaceReady                 = false;
  const QnnInterface_t** providers       = nullptr;
  uint32_t numProviders                  = 0;
  const QnnInterface_t* selectedProvider = nullptr;

  try {
    backendLibraryHandle = dlopen(options.backendPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (backendLibraryHandle == nullptr) {
      throw std::runtime_error("dlopen failed: " + std::string(dlerror()));
    }

    auto getProviders = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
        dlsym(backendLibraryHandle, "QnnInterface_getProviders"));
    if (getProviders == nullptr) {
      throw std::runtime_error("dlsym(QnnInterface_getProviders) failed");
    }

    const auto providerStatus = getProviders(&providers, &numProviders);
    if (providerStatus != QNN_SUCCESS) {
      throw std::runtime_error("QnnInterface_getProviders returned error code " +
                               std::to_string(static_cast<uint64_t>(providerStatus)));
    }
    if (providers == nullptr || numProviders == 0) {
      throw std::runtime_error("no QNN interface providers were returned");
    }

    for (uint32_t i = 0; i < numProviders; ++i) {
      const auto& apiVersion = providers[i]->apiVersion.coreApiVersion;
      if (apiVersion.major == QNN_API_VERSION_MAJOR &&
          apiVersion.minor >= QNN_API_VERSION_MINOR) {
        selectedProvider = providers[i];
        qnnInterface     = providers[i]->QNN_INTERFACE_VER_NAME;
        qnnInterfaceReady = true;
        break;
      }
    }
    if (!qnnInterfaceReady) {
      throw std::runtime_error("unable to find a compatible QNN interface provider");
    }

    if (qnnInterface.logCreate != nullptr) {
      const auto logStatus = qnnInterface.logCreate(qnnLogCallback, logLevel, &logger);
      if (logStatus != QNN_SUCCESS && logStatus != QNN_COMMON_ERROR_NOT_SUPPORTED) {
        throw std::runtime_error("QnnLog_create failed: " + getErrorText(qnnInterface, logStatus));
      }
      if (logStatus == QNN_COMMON_ERROR_NOT_SUPPORTED) {
        logger = nullptr;
      }
    }

    checkQnnStatus(
        qnnInterface, qnnInterface.backendCreate(logger, nullptr, &backend), "QnnBackend_create");

    if (options.dumpBackendInfo) {
      dumpBackendInfo(
          options.backendPath, backendKind, providers, numProviders, selectedProvider, qnnInterface);
    }

    createDeviceWithFallback(
        qnnInterface, logger, backendKind, &device, &platformInfo);

    std::vector<BenchmarkResult> results = {};
    results.reserve(options.precisions.size());
    for (PrecisionKind precision : options.precisions) {
      results.push_back(runBenchmarkCase(qnnInterface,
                                         backend,
                                         device,
                                         backendKind,
                                         precision,
                                         options.rows,
                                         options.inner,
                                         options.cols,
                                         options.warmup,
                                         options.iterations));
    }

    printBenchmarkResults(backendKind, options, results);

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
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << '\n';
    if (qnnInterfaceReady && device != nullptr && qnnInterface.deviceFree != nullptr) {
      qnnInterface.deviceFree(device);
      device = nullptr;
    }
    if (qnnInterfaceReady && platformInfo != nullptr && qnnInterface.deviceFreePlatformInfo != nullptr) {
      qnnInterface.deviceFreePlatformInfo(logger, platformInfo);
      platformInfo = nullptr;
    }
    if (qnnInterfaceReady && backend != nullptr && qnnInterface.backendFree != nullptr) {
      qnnInterface.backendFree(backend);
      backend = nullptr;
    }
    if (qnnInterfaceReady && logger != nullptr && qnnInterface.logFree != nullptr) {
      qnnInterface.logFree(logger);
      logger = nullptr;
    }
    if (backendLibraryHandle != nullptr) {
      dlclose(backendLibraryHandle);
      backendLibraryHandle = nullptr;
    }
    return 1;
  }
}
