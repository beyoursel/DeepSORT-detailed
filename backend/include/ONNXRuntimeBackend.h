#pragma once

#include "IBackend.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

namespace backend {

class ONNXRuntimeBackend : public IBackend {
public:
    ONNXRuntimeBackend() = default;
    ~ONNXRuntimeBackend() override = default;

    bool init(const BackendConfig& cfg) override;
    bool load_model(const std::string& model_path) override;

    bool run(const std::string& input_name,
             const std::vector<float>& input_data,
             const std::vector<int64_t>& input_shape,
             const std::string& output_name,
             std::vector<float>& output_data,
             std::vector<int64_t>& output_shape) override;

private:
    BackendConfig cfg_;
    Ort::Env env_{nullptr};
    Ort::SessionOptions session_options_{nullptr};
    Ort::Session session_{nullptr};

    bool initialized_ = false;
    bool model_loaded_ = false;
};

} // namespace backend
