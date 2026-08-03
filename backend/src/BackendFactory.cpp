#include "BackendFactory.h"
#include "ONNXRuntimeBackend.h"
#ifdef USE_TENSORRT
#include "TensorRTBackend.h"
#endif
#include <stdexcept>

namespace backend {

std::shared_ptr<IBackend> BackendFactory::Create(const BackendConfig& cfg) {
  if (cfg.type == "onnxruntime_cpu" || cfg.type == "onnxruntime_gpu") {
    auto backend = std::make_shared<ONNXRuntimeBackend>();
    if (!backend->Init(cfg)) {
      throw std::runtime_error("Failed to initialize ONNXRuntime backend: " +
                               cfg.type);
    }
    return backend;
  }
  if (cfg.type == "tensorrt") {
#ifdef USE_TENSORRT
    auto backend = std::make_shared<TensorRTBackend>();
    if (!backend->Init(cfg)) {
      throw std::runtime_error("Failed to initialize TensorRT backend");
    }
    return backend;
#else
    throw std::runtime_error(
        "TensorRT backend not compiled in; rebuild with -DUSE_TENSORRT=ON");
#endif
  }
  throw std::runtime_error("Unsupported backend type: " + cfg.type);
}

}  // namespace backend
