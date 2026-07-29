#pragma once

#include "AppConfig.h"
#include "ITracker.h"
#include <memory>

class BYTETracker;  // legacy ByteTrack implementation

// ITracker adapter around the legacy ByteTrack implementation.
class ByteTrackAdapter : public ITracker {
 public:
  explicit ByteTrackAdapter(const ByteTrackConfig& cfg);
  ~ByteTrackAdapter() override;

  std::vector<TrackResult> update(
      const cv::Mat& frame,
      const std::vector<detect_result>& detections) override;

 private:
  std::unique_ptr<BYTETracker> impl_;
};
