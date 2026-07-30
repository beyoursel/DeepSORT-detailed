#pragma once

#include "AppConfig.h"
#include "ITracker.h"
#include <memory>

class Tracker;  // legacy DeepSORT Tracker

namespace tracking {

// ITracker adapter around the legacy DeepSORT implementation.
class DeepSORTAdapter : public ITracker {
 public:
  explicit DeepSORTAdapter(const DeepSORTConfig& cfg);
  ~DeepSORTAdapter() override;

  std::vector<TrackResult> Update(
      const cv::Mat& frame,
      const std::vector<detect_result>& detections) override;

 private:
  std::unique_ptr<::Tracker> impl_;
};

}  // namespace tracking
