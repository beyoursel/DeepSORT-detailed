#pragma once

#include "ITracker.h"
#include <memory>
#include <string>

namespace tracking {

class TrackerFactory {
 public:
  // type: "deepsort" | "bytetrack".
  // Tracker parameters are read from the global AppConfig.
  // Throws std::runtime_error on unknown type.
  static std::unique_ptr<ITracker> Create(const std::string& type);
};

}  // namespace tracking
