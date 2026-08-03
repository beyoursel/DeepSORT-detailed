#pragma once

#include "IBackend.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace backend {

// TensorRT backend: builds an engine from ONNX (with disk cache) and runs
// inference through an IExecutionContext. FP16 only affects internal compute
// precision; network I/O stays FP32, so the IBackend float contract holds.
class TensorRTBackend : public IBackend {
 public:
  TensorRTBackend() = default;
  ~TensorRTBackend() override;

  bool Init(const BackendConfig& cfg) override;
  bool LoadModel(const std::string& model_path) override;

  bool Run(const std::string& input_name, const std::vector<float>& input_data,
           const std::vector<int64_t>& input_shape,
           const std::string& output_name, std::vector<float>& output_data,
           std::vector<int64_t>& output_shape) override;

  std::vector<std::string> GetInputNames() const override {
    return input_names_;
  }
  std::vector<std::string> GetOutputNames() const override {
    return output_names_;
  }

 private:
  bool BuildEngineFromOnnx(const std::string& model_path,
                           const std::string& engine_path);
  bool LoadEngineFromFile(const std::string& engine_path);
  bool EnsureDeviceBuffer(int binding, size_t bytes);

  BackendConfig cfg_;
  cudaStream_t stream_ = nullptr;

  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;

  // Grow-only device buffers, one per binding.
  std::vector<void*> device_buffers_;
  std::vector<size_t> device_buffer_bytes_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  bool initialized_ = false;
  bool model_loaded_ = false;
};

}  // namespace backend
