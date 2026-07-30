#pragma once

#include "AppConfig.h"
#include "ITracker.h"
#include <memory>

class tracker;  // legacy DeepSORT tracker

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
  std::unique_ptr<::tracker> impl_;
};

}  // namespace tracking
