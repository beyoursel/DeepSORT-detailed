#pragma once
#include <vector>

#include "AppConfig.h"
#include "kalmanfilter.h"
#include "model.h"
#include "track.h"

class NearNeighborDisMetric;

class Tracker {
 public:
  NearNeighborDisMetric* metric;
  float max_iou_distance;
  int max_age;
  int n_init;

  KalmanFilter* kf;

  int next_idx;

 public:
  std::vector<Track> tracks;
  explicit Tracker(const DeepSORTConfig& cfg);
  void Predict();
  void Update(const DETECTIONS& detections);
  typedef DYNAMICM (Tracker::*GATED_METRIC_FUNC)(
      std::vector<Track>& tracks, const DETECTIONS& dets,
      const std::vector<int>& track_indices,
      const std::vector<int>& detection_indices);

 private:
  void Match(const DETECTIONS& detections, TRACHER_MATCHD& res);
  void InitiateTrack(const DETECTION_ROW& detection);

 public:
  DYNAMICM GatedMatric(std::vector<Track>& tracks, const DETECTIONS& dets,
                        const std::vector<int>& track_indices,
                        const std::vector<int>& detection_indices);
  DYNAMICM IouCost(std::vector<Track>& tracks, const DETECTIONS& dets,
                    const std::vector<int>& track_indices,
                    const std::vector<int>& detection_indices);
  Eigen::VectorXf Iou(DETECTBOX& bbox, DETECTBOXSS& candidates);
};
