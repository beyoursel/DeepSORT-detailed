#include "BackendFactory.h"
#include "ONNXRuntimeBackend.h"
#include <stdexcept>

namespace backend {

std::shared_ptr<IBackend> BackendFactory::create(const BackendConfig& cfg) {
    if (cfg.type == "onnxruntime_cpu" || cfg.type == "onnxruntime_gpu") {
        auto backend = std::make_shared<ONNXRuntimeBackend>();
        if (!backend->init(cfg)) {
            throw std::runtime_error("Failed to initialize ONNXRuntime backend: " + cfg.type);
        }
        return backend;
    }
    throw std::runtime_error("Unsupported backend type: " + cfg.type);
}

} // namespace backend
