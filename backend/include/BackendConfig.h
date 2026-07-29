#pragma once

#include <string>

namespace backend {

struct BackendConfig {
  std::string type;  // e.g. "onnxruntime_cpu", "onnxruntime_gpu"
  std::string model_path;
  int device_id = 0;  // GPU device id, used when type is "onnxruntime_gpu"
};

}  // namespace backend
