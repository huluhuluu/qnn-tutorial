#include <QNN/QnnInterface.h>
#include <QNN/QnnLog.h>
#include <QNN/QnnOpDef.h>
#include <QNN/QnnTypes.h>
#include <QNN/HTP/QnnHtpDevice.h>

#include <dlfcn.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using QnnInterfaceGetProvidersFn_t =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t*** providerList, uint32_t* numProviders);

struct StageTiming {
  std::string name;
  long long microseconds;
};

struct AppOptions {
  std::string backendPath;
  bool dumpBackendInfo;
};

struct CapabilitySpec {
  const char* name;
  QnnProperty_Key_t key;
};

enum class BackendKind {
  Cpu,
  Gpu,
  Htp,
  Unknown
};

template <typename Fn>
void measureStage(std::vector<StageTiming>& timings, const std::string& name, Fn&& fn) {
  const auto start = Clock::now();
  fn();
  const auto stop = Clock::now();
  timings.push_back(
      {name, std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()});
}

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

void printUsage(const char* programName) {
  std::cout << "Usage: " << programName
            << " [--backend <path>] [--dump-backend-info] [backend_path]\n"
            << "  --backend <path>       Explicit backend shared library path.\n"
            << "  --dump-backend-info    Print provider/interface/capability info.\n"
            << "  backend_path           Positional shorthand for --backend <path>.\n"
            << "  QNN_DUMP_BACKEND_INFO=1 also enables backend info dump.\n";
}

AppOptions parseCommandLine(int argc, char** argv) {
  AppOptions options{defaultBackendPath(), parseBoolEnv("QNN_DUMP_BACKEND_INFO")};
  bool backendPathSet = false;

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
      backendPathSet      = true;
      continue;
    }
    if (arg.rfind("--", 0) == 0) {
      throw std::runtime_error("unknown option: " + arg);
    }
    if (backendPathSet) {
      throw std::runtime_error("multiple backend paths were provided");
    }
    options.backendPath = arg;
    backendPathSet      = true;
  }

  return options;
}

// Infer the runtime backend from the shared library name so one binary can
// switch between CPU, GPU, and HTP without recompilation.
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
    case 5:
      return "DSP";
    case 6:
      return "HTP";
    case 7:
      return "HTA";
    case 9:
      return "IR";
    case 12:
      return "LPAI";
    case 13:
      return "HTP_QEMU";
    case 14:
      return "GENAI_TRANSFORMER";
    case 16:
      return "LPAI_ISLAND";
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

QnnLog_Level_t defaultLogLevel() {
  return QNN_LOG_LEVEL_ERROR;
}

QnnLog_Level_t parseLogLevelFromEnv() {
  const char* raw = std::getenv("QNN_LOG_LEVEL");
  if (raw == nullptr || *raw == '\0') {
    return defaultLogLevel();
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
  return defaultLogLevel();
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

// HTP may run float graphs with FP16 math internally, so its validation
// tolerance is intentionally looser than CPU and GPU.
double validationToleranceForBackend(BackendKind kind) {
  switch (kind) {
    case BackendKind::Htp:
      return 5e-3;
    case BackendKind::Cpu:
    case BackendKind::Gpu:
    case BackendKind::Unknown:
      return 1e-4;
  }
  return 1e-4;
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

Qnn_Tensor_t makeTensor(const char* name,
                        Qnn_TensorType_t type,
                        Qnn_DataType_t dataType,
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

struct ValidationResult {
  double maxAbsError;
  double meanAbsError;
  bool passed;
};

ValidationResult validateOutput(const std::vector<float>& actual,
                                const std::vector<float>& expected,
                                double tolerance) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("output size mismatch");
  }

  double maxAbsError  = 0.0;
  double sumAbsError  = 0.0;
  bool hasInvalidData = false;

  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i])) {
      hasInvalidData = true;
      maxAbsError    = std::numeric_limits<double>::infinity();
      break;
    }
    const double absError = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
    maxAbsError = std::max(maxAbsError, absError);
    sumAbsError += absError;
  }

  const double meanAbsError = hasInvalidData ? std::numeric_limits<double>::infinity()
                                             : (sumAbsError / static_cast<double>(actual.size()));

  return {maxAbsError, meanAbsError, !hasInvalidData && maxAbsError <= tolerance};
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
              << propertyStatusName(status);
    if (status != QNN_PROPERTY_SUPPORTED && status != QNN_PROPERTY_NOT_SUPPORTED &&
        status != QNN_PROPERTY_ERROR_UNKNOWN_KEY) {
      std::cout << " (" << status << ")";
    }
    std::cout << '\n';
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

void printTimingTable(const std::vector<StageTiming>& timings) {
  std::cout << "\nTime Breakdown\n";
  std::cout << std::left << std::setw(28) << "stage" << std::right << std::setw(12) << "time(us)"
            << '\n';
  std::cout << std::string(40, '-') << '\n';
  for (const auto& item : timings) {
    std::cout << std::left << std::setw(28) << item.name << std::right << std::setw(12)
              << item.microseconds << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  constexpr uint32_t kRows      = 32;
  constexpr uint32_t kInner     = 128;
  constexpr uint32_t kCols      = 64;
  constexpr uint32_t kTensorRank = 2;

  const AppOptions options      = parseCommandLine(argc, argv);
  const std::string backendPath = options.backendPath;
  const BackendKind backendKind = detectBackendKind(backendPath);
  const QnnLog_Level_t logLevel = parseLogLevelFromEnv();
  const double validationTolerance = validationToleranceForBackend(backendKind);

  void* backendLibraryHandle       = nullptr;
  Qnn_BackendHandle_t backend      = nullptr;
  Qnn_DeviceHandle_t device        = nullptr;
  Qnn_ContextHandle_t context      = nullptr;
  Qnn_LogHandle_t logger           = nullptr;
  const QnnDevice_PlatformInfo_t* platformInfo = nullptr;
  QNN_INTERFACE_VER_TYPE qnnInterface{};
  bool qnnInterfaceReady           = false;
  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders            = 0;
  const QnnInterface_t* selectedProvider = nullptr;
  std::vector<StageTiming> timings = {};

  try {
    measureStage(timings, "load_backend_library", [&]() {
      backendLibraryHandle = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (backendLibraryHandle == nullptr) {
        throw std::runtime_error("dlopen failed: " + std::string(dlerror()));
      }
    });

    measureStage(timings, "resolve_qnn_interface", [&]() {
      auto getProviders = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
          dlsym(backendLibraryHandle, "QnnInterface_getProviders"));
      if (getProviders == nullptr) {
        throw std::runtime_error("dlsym(QnnInterface_getProviders) failed");
      }

      const auto status = getProviders(&providers, &numProviders);
      if (status != QNN_SUCCESS) {
        throw std::runtime_error("QnnInterface_getProviders returned error code " +
                                 std::to_string(static_cast<uint64_t>(status)));
      }
      if (providers == nullptr || numProviders == 0) {
        throw std::runtime_error("no QNN interface providers were returned");
      }

      bool foundCompatibleProvider = false;
      for (uint32_t i = 0; i < numProviders; ++i) {
        const auto& apiVersion = providers[i]->apiVersion.coreApiVersion;
        if (apiVersion.major == QNN_API_VERSION_MAJOR && apiVersion.minor >= QNN_API_VERSION_MINOR) {
          selectedProvider       = providers[i];
          qnnInterface           = providers[i]->QNN_INTERFACE_VER_NAME;
          qnnInterfaceReady      = true;
          foundCompatibleProvider = true;
          break;
        }
      }

      if (!foundCompatibleProvider) {
        throw std::runtime_error("unable to find a compatible QNN interface provider");
      }
    });

    measureStage(timings, "backend_create", [&]() {
      if (qnnInterface.logCreate != nullptr) {
        const auto logStatus =
            qnnInterface.logCreate(qnnLogCallback, logLevel, &logger);
        if (logStatus != QNN_SUCCESS && logStatus != QNN_COMMON_ERROR_NOT_SUPPORTED) {
          throw std::runtime_error("QnnLog_create failed: " + getErrorText(qnnInterface, logStatus));
        }
        if (logStatus == QNN_COMMON_ERROR_NOT_SUPPORTED) {
          logger = nullptr;
        }
      }

      checkQnnStatus(
          qnnInterface, qnnInterface.backendCreate(logger, nullptr, &backend), "QnnBackend_create");
    });

    if (options.dumpBackendInfo) {
      measureStage(timings, "dump_backend_info", [&]() {
        dumpBackendInfo(
            backendPath, backendKind, providers, numProviders, selectedProvider, qnnInterface);
      });
    }

    measureStage(timings, "device_create", [&]() {
      if (qnnInterface.deviceCreate != nullptr) {
        auto tryDeviceCreate = [&](const QnnDevice_Config_t** configs) {
          device = nullptr;
          return qnnInterface.deviceCreate(logger, configs, &device);
        };

        auto status = tryDeviceCreate(nullptr);
        if (status == QNN_SUCCESS || status == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
          if (status == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
            device = nullptr;
          }
          return;
        }

        if (backendKind == BackendKind::Htp) {
          // Some HTP targets need explicit platform hints before device creation
          // succeeds. Try progressively more explicit configurations.
          if (qnnInterface.deviceGetPlatformInfo != nullptr) {
            const auto infoStatus = qnnInterface.deviceGetPlatformInfo(logger, &platformInfo);
            if (infoStatus == QNN_SUCCESS && platformInfo != nullptr) {
              QnnDevice_Config_t platformConfig = QNN_DEVICE_CONFIG_INIT;
              platformConfig.option       = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
              platformConfig.hardwareInfo = const_cast<QnnDevice_PlatformInfo_t*>(platformInfo);
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
          device = nullptr;
        }
      }
    });

    measureStage(timings, "context_create", [&]() {
      checkQnnStatus(qnnInterface,
                     qnnInterface.contextCreate(backend, device, nullptr, &context),
                     "QnnContext_create");
    });

    Qnn_GraphHandle_t graph = nullptr;
    measureStage(timings, "graph_create", [&]() {
      checkQnnStatus(qnnInterface,
                     qnnInterface.graphCreate(context, "matmul_graph", nullptr, &graph),
                     "QnnGraph_create");
    });

    std::vector<float> inputData  = makeInputData(kRows * kInner, 23, 11.0f);
    std::vector<float> weightData = makeInputData(kInner * kCols, 17, 9.0f);
    std::vector<float> outputData(kRows * kCols, std::numeric_limits<float>::quiet_NaN());
    const std::vector<float> expected = referenceMatmul(inputData, weightData, kRows, kInner, kCols);

    std::array<uint32_t, kTensorRank> inputDims  = {kRows, kInner};
    std::array<uint32_t, kTensorRank> weightDims = {kInner, kCols};
    std::array<uint32_t, kTensorRank> outputDims = {kRows, kCols};

    // App I/O tensors are registered as metadata first; real buffers are bound
    // only when graphExecute() is called.
    Qnn_Tensor_t inputTensor = makeTensor("input",
                                          QNN_TENSOR_TYPE_APP_WRITE,
                                          QNN_DATATYPE_FLOAT_32,
                                          inputDims.data(),
                                          kTensorRank,
                                          nullptr,
                                          0);
    Qnn_Tensor_t weightTensor = makeTensor("weight",
                                           QNN_TENSOR_TYPE_STATIC,
                                           QNN_DATATYPE_FLOAT_32,
                                           weightDims.data(),
                                           kTensorRank,
                                           weightData.data(),
                                           static_cast<uint32_t>(weightData.size() * sizeof(float)));
    Qnn_Tensor_t outputTensor = makeTensor("output",
                                           QNN_TENSOR_TYPE_APP_READ,
                                           QNN_DATATYPE_FLOAT_32,
                                           outputDims.data(),
                                           kTensorRank,
                                           nullptr,
                                           0);

    measureStage(timings, "tensor_create", [&]() {
      checkQnnStatus(qnnInterface,
                     qnnInterface.tensorCreateGraphTensor(graph, &inputTensor),
                     "QnnTensor_createGraphTensor(input)");
      checkQnnStatus(qnnInterface,
                     qnnInterface.tensorCreateGraphTensor(graph, &weightTensor),
                     "QnnTensor_createGraphTensor(weight)");
      checkQnnStatus(qnnInterface,
                     qnnInterface.tensorCreateGraphTensor(graph, &outputTensor),
                     "QnnTensor_createGraphTensor(output)");
    });

    std::array<Qnn_Tensor_t, 2> nodeInputs = {inputTensor, weightTensor};
    std::array<Qnn_Tensor_t, 1> nodeOutputs = {outputTensor};

    // Build a single MatMul node: output = input x static weight.
    Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
    opConfig.v1.name          = "matmul_0";
    opConfig.v1.packageName   = QNN_OP_PACKAGE_NAME_QTI_AISW;
    opConfig.v1.typeName      = QNN_OP_MAT_MUL;
    opConfig.v1.numOfParams   = 0;
    opConfig.v1.params        = nullptr;
    opConfig.v1.numOfInputs   = static_cast<uint32_t>(nodeInputs.size());
    opConfig.v1.inputTensors  = nodeInputs.data();
    opConfig.v1.numOfOutputs  = static_cast<uint32_t>(nodeOutputs.size());
    opConfig.v1.outputTensors = nodeOutputs.data();

    measureStage(timings, "graph_add_node", [&]() {
      checkQnnStatus(
          qnnInterface, qnnInterface.graphAddNode(graph, opConfig), "QnnGraph_addNode(MatMul)");
    });

    measureStage(timings, "graph_finalize", [&]() {
      checkQnnStatus(qnnInterface,
                     qnnInterface.graphFinalize(graph, nullptr, nullptr),
                     "QnnGraph_finalize");
    });

    // Reuse the registered tensor descriptors and attach concrete host buffers
    // for this inference run.
    Qnn_Tensor_t executeInput = makeExecuteTensor(
        inputTensor, inputData.data(), static_cast<uint32_t>(inputData.size() * sizeof(float)));
    Qnn_Tensor_t executeOutput = makeExecuteTensor(
        outputTensor, outputData.data(), static_cast<uint32_t>(outputData.size() * sizeof(float)));
    Qnn_Tensor_t executeInputs[]  = {executeInput};
    Qnn_Tensor_t executeOutputs[] = {executeOutput};

    measureStage(timings, "graph_execute", [&]() {
      checkQnnStatus(qnnInterface,
                     qnnInterface.graphExecute(graph,
                                               executeInputs,
                                               1,
                                               executeOutputs,
                                               1,
                                               nullptr,
                                               nullptr),
                     "QnnGraph_execute");
    });

    ValidationResult validation{};
    measureStage(timings, "result_validate", [&]() {
      validation = validateOutput(outputData, expected, validationTolerance);
    });

    std::cout << "QNN online MatMul example\n";
    std::cout << "backend_path : " << backendPath << '\n';
    std::cout << "backend_kind : " << backendKindName(backendKind) << '\n';
    if (backendKind == BackendKind::Htp) {
      std::cout << "htp_soc_model: " << defaultHtpSocModel() << '\n';
      std::cout << "htp_arch     : " << static_cast<int>(defaultHtpArch()) << '\n';
    }
    std::cout << "tolerance    : " << validationTolerance << '\n';
    std::cout << "shape        : A[" << kRows << ", " << kInner << "] x B[" << kInner << ", "
              << kCols << "] -> C[" << kRows << ", " << kCols << "]\n";
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "max_abs_err  : " << validation.maxAbsError << '\n';
    std::cout << "mean_abs_err : " << validation.meanAbsError << '\n';
    std::cout << "validation   : " << (validation.passed ? "PASS" : "FAIL") << '\n';
    std::cout << "output[0:8]  :";
    for (size_t i = 0; i < 8 && i < outputData.size(); ++i) {
      std::cout << ' ' << outputData[i];
    }
    std::cout << '\n';

    printTimingTable(timings);

    if (context != nullptr) {
      checkQnnStatus(qnnInterface,
                     qnnInterface.contextFree(context, nullptr),
                     "QnnContext_free");
      context = nullptr;
    }

    if (device != nullptr && qnnInterface.deviceFree != nullptr) {
      checkQnnStatus(qnnInterface, qnnInterface.deviceFree(device), "QnnDevice_free");
      device = nullptr;
    }

    if (platformInfo != nullptr && qnnInterface.deviceFreePlatformInfo != nullptr) {
      checkQnnStatus(
          qnnInterface, qnnInterface.deviceFreePlatformInfo(logger, platformInfo), "QnnDevice_freePlatformInfo");
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

    return validation.passed ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << '\n';

    if (qnnInterfaceReady && context != nullptr && qnnInterface.contextFree != nullptr) {
      qnnInterface.contextFree(context, nullptr);
      context = nullptr;
    }
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
