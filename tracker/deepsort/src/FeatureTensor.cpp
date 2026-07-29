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

FeatureTensor* FeatureTensor::instance = NULL;

FeatureTensor* FeatureTensor::getInstance() {
  if (instance == NULL) {
    instance = new FeatureTensor();
  }
  return instance;
}

FeatureTensor::FeatureTensor() {
  const DeepSORTConfig& cfg = AppConfig::getInstance()->deepsort;

  results_.resize(cfg.feature_dim);
  output_shape_ = {1, cfg.feature_dim};

  backend::BackendConfig backend_cfg;
  backend_cfg.type = cfg.backend;
  backend_cfg.model_path = cfg.feature_model_path;
  backend_cfg.device_id = AppConfig::getInstance()->backend.device_id;

  backend_ = backend::BackendFactory::create(backend_cfg);
  if (!backend_->load_model(backend_cfg.model_path)) {
    throw std::runtime_error("FeatureTensor init failed: cannot load model " +
                             backend_cfg.model_path);
  }

  // prepare model:
  if (!init()) {
    throw std::runtime_error(
        "FeatureTensor init failed: dummy inference failed");
  }
  std::cout << "FeatureTensor init succeed" << std::endl;
}

FeatureTensor::~FeatureTensor() {}

bool FeatureTensor::init() {
  // Query input shape from backend by running a dummy inference.
  std::vector<float> dummy_input(width_ * height_ * 3, 0.0f);
  std::vector<float> dummy_output;
  std::vector<int64_t> output_shape;

  if (!backend_->run("input", dummy_input, input_shape_, "output", dummy_output,
                     output_shape)) {
    std::cerr << "FeatureTensor init failed: dummy inference failed"
              << std::endl;
    return false;
  }

  inputDims_ = input_shape_;
  std::cout << "Input Dimensions: " << inputDims_
            << std::endl;  // [1, 3, 128, 64]
  std::cout << "FeatureTensor::init() " << std::endl;

  return true;
}

void FeatureTensor::preprocess(cv::Mat& imageBGR,
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
  inputTensorSize = vectorProduct(inputDims_);
  inputTensorValues.assign(preprocessedImage.begin<float>(),
                           preprocessedImage.end<float>());
}

bool FeatureTensor::getRectsFeature(const cv::Mat& img, DETECTIONS& d) {
  ScopedTimer timer("reid");

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
    preprocess(mattmp, inputTensorValues, inputTensorSize);

    if (!backend_->run("input", inputTensorValues, inputDims_, "output",
                       results_, output_shape_)) {
      std::cerr << "FeatureTensor inference failed" << std::endl;
      return false;
    }

    dbox.feature.resize(1, output_shape_[1]);
    for (int i = 0; i < output_shape_[1]; i++) {
      dbox.feature[i] = results_[i];
    }
  }

  return true;
}

void FeatureTensor::tobuffer(const std::vector<cv::Mat>& imgs, uint8* buf) {
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
void FeatureTensor::test() { return; }
