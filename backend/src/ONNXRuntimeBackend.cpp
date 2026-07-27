#include "ONNXRuntimeBackend.h"
#include <cstring>
#include <iostream>

namespace backend {

bool ONNXRuntimeBackend::init(const BackendConfig& cfg) {
    cfg_ = cfg;

    try {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeBackend");
        session_options_ = Ort::SessionOptions();
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg_.type == "onnxruntime_gpu") {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = cfg_.device_id;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "Using ONNXRuntime CUDA provider, device_id=" << cfg_.device_id << std::endl;
        } else {
            std::cout << "Using ONNXRuntime CPU provider" << std::endl;
        }

        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ONNXRuntimeBackend::init failed: " << e.what() << std::endl;
        return false;
    }
}

bool ONNXRuntimeBackend::load_model(const std::string& model_path) {
    if (!initialized_) {
        std::cerr << "Backend not initialized" << std::endl;
        return false;
    }
    try {
        session_ = Ort::Session(env_, model_path.c_str(), session_options_);
        model_loaded_ = true;
        std::cout << "Loaded model: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        model_loaded_ = false;
        std::cerr << "Failed to load model " << model_path << ": " << e.what() << std::endl;
        return false;
    }
}

bool ONNXRuntimeBackend::run(const std::string& input_name,
                             const std::vector<float>& input_data,
                             const std::vector<int64_t>& input_shape,
                             const std::string& output_name,
                             std::vector<float>& output_data,
                             std::vector<int64_t>& output_shape) {
    if (!initialized_) {
        std::cerr << "Backend not initialized" << std::endl;
        return false;
    }
    if (!model_loaded_) {
        std::cerr << "Model not loaded, cannot run inference" << std::endl;
        return false;
    }

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // ONNXRuntime's CreateTensor takes a non-const float* buffer. Copy the
        // const input data into a local writable buffer so we don't break the
        // IBackend interface contract.
        std::vector<float> input_buffer(input_data);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_buffer.data(), input_buffer.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = {input_name.c_str()};
        const char* output_names[] = {output_name.c_str()};

        Ort::RunOptions run_options;
        std::vector<Ort::Value> output_tensors = session_.Run(
            run_options, input_names, &input_tensor, 1, output_names, 1);

        Ort::Value& output_tensor = output_tensors.front();
        Ort::TensorTypeAndShapeInfo shape_info = output_tensor.GetTensorTypeAndShapeInfo();
        output_shape = shape_info.GetShape();

        size_t output_size = 1;
        for (int64_t dim : output_shape) {
            output_size *= static_cast<size_t>(dim);
        }

        output_data.resize(output_size);
        const float* raw_output = output_tensor.GetTensorData<float>();
        std::memcpy(output_data.data(), raw_output, output_size * sizeof(float));

        return true;
    } catch (const std::exception& e) {
        std::cerr << "ONNXRuntimeBackend::run failed: " << e.what() << std::endl;
        return false;
    }
}

} // namespace backend
