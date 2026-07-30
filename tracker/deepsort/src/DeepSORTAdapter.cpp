#include "DeepSORTAdapter.h"
#include "FeatureTensor.h"
#include "Timer.h"
#include "tracker.h"

namespace tracking {

DeepSORTAdapter::DeepSORTAdapter(const DeepSORTConfig& cfg)
    : impl_(new ::Tracker(cfg)) {}

DeepSORTAdapter::~DeepSORTAdapter() = default;

std::vector<TrackResult> DeepSORTAdapter::Update(
    const cv::Mat& frame, const std::vector<detect_result>& dets) {
  ScopedTimer timer("track");

  DETECTIONS detections;
  for (const detect_result& dr : dets) {
    DETECTION_ROW row;
    row.tlwh = DETECTBOX(dr.box.x, dr.box.y, dr.box.width, dr.box.height);
    row.confidence = dr.confidence;
    detections.push_back(row);
  }

  std::vector<TrackResult> out;
  // May throw on ReID model load failure (first call); let the caller handle
  // it.
  if (!FeatureTensor::GetInstance()->GetRectsFeature(frame, detections)) {
    return out;
  }

  impl_->Predict();
  impl_->Update(detections);

  for (Track& track : impl_->tracks) {
    if (!track.IsConfirmed() || track.time_since_update > 1) continue;
    DETECTBOX tlwh = track.ToTlwh();
    out.push_back(
        {track.track_id, cv::Rect_<float>(tlwh(0), tlwh(1), tlwh(2), tlwh(3))});
  }
  return out;
}

}  // namespace tracking
