#pragma once

#include <string>

namespace backend {

struct BackendConfig {
  std::string type;  // e.g. "onnxruntime_cpu", "onnxruntime_gpu", "tensorrt"
  std::string model_path;
  int device_id = 0;   // GPU device id, used by GPU backends
  bool fp16 = true;    // tensorrt only: build engine with FP16 compute
};

}  // namespace backend
