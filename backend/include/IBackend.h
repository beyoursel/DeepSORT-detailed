#pragma once

#include "BackendConfig.h"
#include <string>
#include <vector>

namespace backend {

class IBackend {
 public:
  virtual ~IBackend() = default;

  virtual bool init(const BackendConfig& cfg) = 0;
  virtual bool load_model(const std::string& model_path) = 0;

  // Run inference.
  // input_name/output_name: ONNX node names (ignored by non-ONNX backends).
  // input_data: flattened input tensor in NCHW order.
  // input_shape: shape of input tensor.
  // output_data: flattened output tensor (resized by backend).
  // output_shape: shape of output tensor.
  virtual bool run(const std::string& input_name,
                   const std::vector<float>& input_data,
                   const std::vector<int64_t>& input_shape,
                   const std::string& output_name,
                   std::vector<float>& output_data,
                   std::vector<int64_t>& output_shape) = 0;
};

}  // namespace backend
