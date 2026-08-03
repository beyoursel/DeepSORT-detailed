/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-21 04:32:26
*/

#include "FeatureTensor.h"
#include "AppConfig.h"
#include "BackendFactory.h"
#include "Timer.h"
#include <iostream>

FeatureTensor& FeatureTensor::GetInstance() {
  static FeatureTensor instance;  // thread-safe since C++11
  return instance;
}

FeatureTensor::FeatureTensor() {
  const DeepSORTConfig& cfg = AppConfig::GetInstance().deepsort;

  results_.resize(cfg.feature_dim);
  output_shape_ = {1, cfg.feature_dim};

  backend::BackendConfig backend_cfg;
  backend_cfg.type = cfg.backend;
  backend_cfg.model_path = cfg.feature_model_path;
  backend_cfg.device_id = AppConfig::GetInstance().backend.device_id;
  backend_cfg.fp16 = AppConfig::GetInstance().backend.fp16;

  backend_ = backend::BackendFactory::Create(backend_cfg);
  if (!backend_->LoadModel(backend_cfg.model_path)) {
    throw std::runtime_error("FeatureTensor Init failed: cannot load model " +
                             backend_cfg.model_path);
  }

  // prepare model:
  if (!Init()) {
    throw std::runtime_error(
        "FeatureTensor Init failed: dummy inference failed");
  }
  std::cout << "FeatureTensor Init succeed" << std::endl;
}

FeatureTensor::~FeatureTensor() {}

bool FeatureTensor::Init() {
  // Warm up every batch bucket with a dummy inference. cuDNN runs a one-time
  // (~300 ms) algorithm search for each new input shape, so GetRectsFeature
  // pads the batch to the next power of two and we pre-pay that search here
  // instead of stalling mid-video whenever the crowd size reaches a new max.
  for (int64_t batch : {1, 2, 4, 8, 16, 32, 64}) {
    std::vector<int64_t> shape = input_shape_;
    shape[0] = batch;
    std::vector<float> dummy_input(batch * width_ * height_ * 3, 0.0f);
    std::vector<float> dummy_output;
    std::vector<int64_t> output_shape;

    if (!backend_->Run("input", dummy_input, shape, "output", dummy_output,
                       output_shape)) {
      std::cerr << "FeatureTensor Init failed: dummy inference failed"
                << std::endl;
      return false;
    }
  }

  inputDims_ = input_shape_;
  std::cout << "Input Dimensions: " << inputDims_
            << std::endl;  // [1, 3, 128, 64]
  std::cout << "FeatureTensor::Init() " << std::endl;

  return true;
}

void FeatureTensor::Preprocess(cv::Mat& imageBGR,
                               std::vector<float>& inputTensorValues,
                               size_t& inputTensorSize) {
  // pre-processing the Image
  //  step 1: Read an image in HWC BGR UINT8 format.
  //  cv::Mat imageBGR = cv::imread(imageFilepath,
  //  cv::ImreadModes::IMREAD_COLOR);

  // step 2: Resize the image.
  cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
  cv::resize(imageBGR, resizedImageBGR,
             cv::Size(inputDims_.at(3), inputDims_.at(2)),
             cv::InterpolationFlags::INTER_CUBIC);

  // cv::resize(imageBGR, resizedImageBGR,
  //            cv::Size(64, 128));

  // step 3: Convert the image to HWC RGB UINT8 format.
  cv::cvtColor(resizedImageBGR, resizedImageRGB,
               cv::ColorConversionCodes::COLOR_BGR2RGB);
  // step 4: Convert the image to HWC RGB float format by dividing each pixel by
  // 255.
  resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

  // step 5: Split the RGB channels from the image.
  cv::Mat channels[3];
  cv::split(resizedImage, channels);

  // step 6: Normalize each channel.
  //  Normalization per channel
  //  Normalization parameters obtained from your custom model

  channels[0] = (channels[0] - 0.485) / 0.229;
  channels[1] = (channels[1] - 0.456) / 0.224;
  channels[2] = (channels[2] - 0.406) / 0.225;

  // step 7: Merge the RGB channels back to the image.
  cv::merge(channels, 3, resizedImage);

  // step 8: Convert the image to CHW RGB float format.
  // HWC to CHW
  cv::dnn::blobFromImage(resizedImage, preprocessedImage);
  inputTensorSize = VectorProduct(inputDims_);
  inputTensorValues.assign(preprocessedImage.begin<float>(),
                           preprocessedImage.end<float>());
}

namespace {
// Round up to the next power of two: batch sizes are bucketed so cuDNN only
// ever sees the shapes warmed up in FeatureTensor::Init().
size_t NextPow2(size_t n) {
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}
}  // namespace

bool FeatureTensor::GetRectsFeature(const cv::Mat& img, DETECTIONS& d) {
  ScopedTimer timer("reid");

  if (d.empty()) return true;

  // Batched inference: preprocess every detection crop with the exact same
  // per-crop pipeline as before (INTER_CUBIC resize + ImageNet mean/std),
  // concatenate into one (N, 3, 128, 64) tensor and call Run() once.
  const size_t per_crop_size = VectorProduct(inputDims_);
  std::vector<float> batch_input;
  batch_input.reserve(d.size() * per_crop_size);

  for (DETECTION_ROW& dbox : d) {
    cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                           int(dbox.tlwh(2)), int(dbox.tlwh(3)));
    rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
    rc.width = rc.height * 0.5;
    rc.x = (rc.x >= 0 ? rc.x : 0);
    rc.y = (rc.y >= 0 ? rc.y : 0);
    rc.width = (rc.x + rc.width <= img.cols ? rc.width : (img.cols - rc.x));
    rc.height = (rc.y + rc.height <= img.rows ? rc.height : (img.rows - rc.y));

    cv::Mat mattmp = img(rc).clone();

    std::vector<float> inputTensorValues;
    size_t inputTensorSize;
    Preprocess(mattmp, inputTensorValues, inputTensorSize);
    batch_input.insert(batch_input.end(), inputTensorValues.begin(),
                       inputTensorValues.end());
  }

  // Pad with zero rows up to the bucket size; padded rows produce garbage
  // features that are simply not read back below.
  const size_t batch_size = d.size();
  const size_t padded_size = NextPow2(batch_size);
  batch_input.resize(padded_size * per_crop_size, 0.0f);

  std::vector<int64_t> batch_input_shape = inputDims_;
  batch_input_shape[0] = static_cast<int64_t>(padded_size);
  std::vector<float> batch_output;
  std::vector<int64_t> batch_output_shape;
  if (!backend_->Run("input", batch_input, batch_input_shape, "output",
                     batch_output, batch_output_shape)) {
    std::cerr << "FeatureTensor inference failed" << std::endl;
    return false;
  }

  const int64_t feature_dim = batch_output_shape[1];
  for (size_t i = 0; i < batch_size; i++) {
    d[i].feature.resize(1, feature_dim);
    for (int64_t j = 0; j < feature_dim; j++) {
      d[i].feature[j] = batch_output[i * feature_dim + j];
    }
  }

  return true;
}

void FeatureTensor::ToBuffer(const std::vector<cv::Mat>& imgs, uint8* buf) {
  int pos = 0;
  for (const cv::Mat& img : imgs) {
    int Lenth = img.rows * img.cols * 3;
    int nr = img.rows;
    int nc = img.cols;
    if (img.isContinuous()) {
      nr = 1;
      nc = Lenth;
    }
    for (int i = 0; i < nr; i++) {
      const uchar* inData = img.ptr<uchar>(i);
      for (int j = 0; j < nc; j++) {
        buf[pos] = *inData++;
        pos++;
      }
    }  // end for
  }    // end imgs;
}
void FeatureTensor::Test() { return; }
