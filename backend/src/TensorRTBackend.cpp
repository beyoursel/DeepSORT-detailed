#include "TensorRTBackend.h"
#include <NvOnnxParser.h>
#include <cstring>
#include <fstream>
#include <iostream>

namespace backend {

namespace {

class TrtLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << msg << std::endl;
    }
  }
};

TrtLogger g_trt_logger;

nvinfer1::Dims ToDims(const std::vector<int64_t>& shape) {
  nvinfer1::Dims dims;
  dims.nbDims = static_cast<int>(shape.size());
  for (int i = 0; i < dims.nbDims; ++i) {
    dims.d[i] = shape[i];
  }
  return dims;
}

size_t Volume(const nvinfer1::Dims& dims) {
  size_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    v *= static_cast<size_t>(dims.d[i]);
  }
  return v;
}

}  // namespace

TensorRTBackend::~TensorRTBackend() {
  for (void* buf : device_buffers_) {
    if (buf) cudaFree(buf);
  }
  if (stream_) cudaStreamDestroy(stream_);
  if (context_) context_->destroy();
  if (engine_) engine_->destroy();
  if (runtime_) runtime_->destroy();
}

bool TensorRTBackend::Init(const BackendConfig& cfg) {
  cfg_ = cfg;
  if (cudaSetDevice(cfg_.device_id) != cudaSuccess) {
    std::cerr << "TensorRTBackend::Init failed: cudaSetDevice "
              << cfg_.device_id << std::endl;
    return false;
  }
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    std::cerr << "TensorRTBackend::Init failed: cudaStreamCreate" << std::endl;
    return false;
  }
  initialized_ = true;
  std::cout << "Using TensorRT backend, device_id=" << cfg_.device_id
            << ", fp16=" << (cfg_.fp16 ? "on" : "off") << std::endl;
  return true;
}

bool TensorRTBackend::LoadModel(const std::string& model_path) {
  if (!initialized_) {
    std::cerr << "Backend not initialized" << std::endl;
    return false;
  }

  // Engine cache: bound to GPU architecture and TensorRT version; delete
  // *.engine files after upgrading either.
  const std::string engine_path =
      model_path + (cfg_.fp16 ? ".fp16.engine" : ".fp32.engine");

  std::ifstream cache(engine_path, std::ios::binary);
  bool ok = cache.good() ? LoadEngineFromFile(engine_path)
                         : BuildEngineFromOnnx(model_path, engine_path);
  if (!ok) {
    model_loaded_ = false;
    return false;
  }

  context_ = engine_->createExecutionContext();
  if (!context_) {
    std::cerr << "Failed to create TensorRT execution context" << std::endl;
    model_loaded_ = false;
    return false;
  }

  input_names_.clear();
  output_names_.clear();
  const int nb_bindings = engine_->getNbBindings();
  device_buffers_.assign(nb_bindings, nullptr);
  device_buffer_bytes_.assign(nb_bindings, 0);
  for (int i = 0; i < nb_bindings; ++i) {
    if (engine_->bindingIsInput(i)) {
      input_names_.push_back(engine_->getBindingName(i));
    } else {
      output_names_.push_back(engine_->getBindingName(i));
    }
  }

  model_loaded_ = true;
  std::cout << "Loaded model: " << model_path << std::endl;
  return true;
}

bool TensorRTBackend::LoadEngineFromFile(const std::string& engine_path) {
  std::ifstream file(engine_path,
                     std::ios::binary | std::ios::ate);
  if (!file.good()) return false;
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<size_t>(size));
  if (!file.read(data.data(), size)) return false;
  file.close();

  runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
  if (!runtime_) return false;
  engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
  if (!engine_) {
    std::cerr << "Failed to deserialize engine: " << engine_path
              << " (delete stale *.engine caches and retry)" << std::endl;
    return false;
  }
  std::cout << "Loaded cached engine: " << engine_path << std::endl;
  return true;
}

bool TensorRTBackend::BuildEngineFromOnnx(const std::string& model_path,
                                          const std::string& engine_path) {
  std::cout << "Building TensorRT engine from " << model_path
            << " (one-time cost, will be cached)..." << std::endl;

  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_trt_logger);
  if (!builder) return false;

  const auto explicit_batch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition* network =
      builder->createNetworkV2(explicit_batch);
  nvonnxparser::IParser* parser =
      nvonnxparser::createParser(*network, g_trt_logger);
  if (!parser->parseFromFile(
          model_path.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::cerr << "TensorRT ONNX parse failed: " << model_path << std::endl;
    parser->destroy();
    network->destroy();
    builder->destroy();
    return false;
  }

  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  if (cfg_.fp16 && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  // TRT 8.5 defaults to an unbounded workspace pool, which spams
  // OutOfMemory tactic warnings on small GPUs; cap it at 1 GiB.
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             1ULL << 30);

  // Dynamic dims (e.g. ReID batch) need an optimization profile. The max
  // batch of 64 matches the bucket warmup in FeatureTensor::Init().
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  bool has_dynamic = false;
  for (int i = 0; i < network->getNbInputs(); ++i) {
    nvinfer1::ITensor* input = network->getInput(i);
    nvinfer1::Dims dims = input->getDimensions();
    nvinfer1::Dims min_dims = dims, opt_dims = dims, max_dims = dims;
    bool input_dynamic = false;
    for (int d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] == -1) {
        input_dynamic = true;
        min_dims.d[d] = 1;
        opt_dims.d[d] = 8;
        max_dims.d[d] = 64;
      }
    }
    if (input_dynamic) {
      has_dynamic = true;
      profile->setDimensions(input->getName(),
                             nvinfer1::OptProfileSelector::kMIN, min_dims);
      profile->setDimensions(input->getName(),
                             nvinfer1::OptProfileSelector::kOPT, opt_dims);
      profile->setDimensions(input->getName(),
                             nvinfer1::OptProfileSelector::kMAX, max_dims);
    }
  }
  if (has_dynamic) {
    config->addOptimizationProfile(profile);
  }

  nvinfer1::IHostMemory* serialized =
      builder->buildSerializedNetwork(*network, *config);
  if (!serialized) {
    std::cerr << "TensorRT engine build failed: " << model_path << std::endl;
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
    return false;
  }

  std::ofstream out(engine_path, std::ios::binary);
  out.write(static_cast<const char*>(serialized->data()), serialized->size());
  out.close();
  std::cout << "Engine cached to: " << engine_path << std::endl;

  runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
  engine_ =
      runtime_ ? runtime_->deserializeCudaEngine(serialized->data(),
                                                 serialized->size())
               : nullptr;

  serialized->destroy();
  config->destroy();
  parser->destroy();
  network->destroy();
  builder->destroy();

  if (!engine_) {
    std::cerr << "Failed to deserialize freshly built engine" << std::endl;
    return false;
  }
  return true;
}

bool TensorRTBackend::EnsureDeviceBuffer(int binding, size_t bytes) {
  if (device_buffer_bytes_[binding] >= bytes) return true;
  if (device_buffers_[binding]) cudaFree(device_buffers_[binding]);
  if (cudaMalloc(&device_buffers_[binding], bytes) != cudaSuccess) {
    std::cerr << "TensorRT cudaMalloc failed: " << bytes << " bytes"
              << std::endl;
    device_buffers_[binding] = nullptr;
    device_buffer_bytes_[binding] = 0;
    return false;
  }
  device_buffer_bytes_[binding] = bytes;
  return true;
}

bool TensorRTBackend::Run(const std::string& input_name,
                          const std::vector<float>& input_data,
                          const std::vector<int64_t>& input_shape,
                          const std::string& output_name,
                          std::vector<float>& output_data,
                          std::vector<int64_t>& output_shape) {
  if (!initialized_ || !model_loaded_) {
    std::cerr << "TensorRT backend not ready" << std::endl;
    return false;
  }

  const int in_idx = engine_->getBindingIndex(input_name.c_str());
  const int out_idx = engine_->getBindingIndex(output_name.c_str());
  if (in_idx < 0 || out_idx < 0) {
    std::cerr << "TensorRT binding not found: " << input_name << " / "
              << output_name << std::endl;
    return false;
  }

  if (!context_->setBindingDimensions(in_idx, ToDims(input_shape))) {
    std::cerr << "TensorRT setBindingDimensions failed (input shape outside "
                 "the optimization profile?)"
              << std::endl;
    return false;
  }

  const size_t in_bytes = input_data.size() * sizeof(float);
  if (!EnsureDeviceBuffer(in_idx, in_bytes)) return false;
  cudaMemcpyAsync(device_buffers_[in_idx], input_data.data(), in_bytes,
                  cudaMemcpyHostToDevice, stream_);

  const nvinfer1::Dims out_dims = context_->getBindingDimensions(out_idx);
  const size_t out_size = Volume(out_dims);
  if (!EnsureDeviceBuffer(out_idx, out_size * sizeof(float))) return false;

  if (!context_->enqueueV2(device_buffers_.data(), stream_, nullptr)) {
    std::cerr << "TensorRT enqueueV2 failed" << std::endl;
    return false;
  }

  output_data.resize(out_size);
  cudaMemcpyAsync(output_data.data(), device_buffers_[out_idx],
                  out_size * sizeof(float), cudaMemcpyDeviceToHost, stream_);
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    std::cerr << "TensorRT stream sync failed" << std::endl;
    return false;
  }

  output_shape.assign(out_dims.d, out_dims.d + out_dims.nbDims);
  return true;
}

}  // namespace backend
