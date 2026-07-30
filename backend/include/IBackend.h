#pragma once

#include "BackendConfig.h"
#include <string>
#include <vector>

namespace backend {

class IBackend {
 public:
  virtual ~IBackend() = default;

  virtual bool Init(const BackendConfig& cfg) = 0;
  virtual bool LoadModel(const std::string& model_path) = 0;

  // Run inference.
  // input_name/output_name: ONNX node names (ignored by non-ONNX backends).
  // input_data: flattened input tensor in NCHW order.
  // input_shape: shape of input tensor.
  // output_data: flattened output tensor (resized by backend).
  // output_shape: shape of output tensor.
  virtual bool Run(const std::string& input_name,
                   const std::vector<float>& input_data,
                   const std::vector<int64_t>& input_shape,
                   const std::string& output_name,
                   std::vector<float>& output_data,
                   std::vector<int64_t>& output_shape) = 0;

  // Node names of the loaded model; valid only after LoadModel succeeds.
  // Callers should use these instead of hardcoding ONNX node names, which
  // differ across export toolchains (e.g. "output" vs "output0").
  virtual std::vector<std::string> GetInputNames() const = 0;
  virtual std::vector<std::string> GetOutputNames() const = 0;
};

}  // namespace backend
