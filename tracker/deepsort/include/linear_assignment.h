#pragma once
#include "dataType.h"
#include "tracker.h"

#define INFTY_COST 1e5
class Tracker;
// for matching;
class LinearAssignment {
  LinearAssignment();
  LinearAssignment(const LinearAssignment&);
  LinearAssignment& operator=(const LinearAssignment&);
  static LinearAssignment* instance_;

 public:
  static LinearAssignment* GetInstance();
  TRACHER_MATCHD MatchingCascade(
      Tracker* distance_metric, Tracker::GATED_METRIC_FUNC distance_metric_func,
      float max_distance, int cascade_depth, std::vector<Track>& tracks,
      const DETECTIONS& detections, std::vector<int>& track_indices,
      std::vector<int> detection_indices = std::vector<int>());
  TRACHER_MATCHD MinCostMatching(
      Tracker* distance_metric, Tracker::GATED_METRIC_FUNC distance_metric_func,
      float max_distance, std::vector<Track>& tracks,
      const DETECTIONS& detections, std::vector<int>& track_indices,
      std::vector<int>& detection_indices);
  DYNAMICM GateCostMatrix(KalmanFilter* kf, DYNAMICM& cost_matrix,
                            std::vector<Track>& tracks,
                            const DETECTIONS& detections,
                            const std::vector<int>& track_indices,
                            const std::vector<int>& detection_indices,
                            float gated_cost = INFTY_COST,
                            bool only_position = false);
};
