
/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-21 02:39:47
*/
#pragma once

#include "IBackend.h"
#include "dataType.h"
#include "model.h"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <vector>

typedef unsigned char uint8;

template <typename T>
T VectorProduct(const std::vector<T>& v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i != v.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

class FeatureTensor {
 public:
  static FeatureTensor* GetInstance();
  bool GetRectsFeature(const cv::Mat& img, DETECTIONS& d);
  void Preprocess(cv::Mat& imageBGR, std::vector<float>& inputTensorValues,
                  size_t& inputTensorSize);

 private:
  FeatureTensor();
  FeatureTensor(const FeatureTensor&);
  FeatureTensor& operator=(const FeatureTensor&);
  static FeatureTensor* instance_;
  bool Init();
  ~FeatureTensor();

  void ToBuffer(const std::vector<cv::Mat>& imgs, uint8* buf);

 public:
  void Test();

  static constexpr const int width_ = 64;
  static constexpr const int height_ = 128;

  std::array<float, width_ * height_> input_image_{};

  std::vector<float> results_;

  std::shared_ptr<backend::IBackend> backend_;
  std::vector<int64_t> input_shape_{1, 3, height_, width_};
  std::vector<int64_t> output_shape_;

  std::vector<int64_t> inputDims_;
};
