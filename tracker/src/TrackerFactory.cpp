#include "TrackerFactory.h"
#include "AppConfig.h"
#include "ByteTrackAdapter.h"
#include "DeepSORTAdapter.h"
#include <stdexcept>

namespace tracking {

std::unique_ptr<ITracker> TrackerFactory::Create(const std::string& type) {
  AppConfig& cfg = AppConfig::GetInstance();
  if (type == "deepsort") {
    return std::unique_ptr<ITracker>(new DeepSORTAdapter(cfg.deepsort));
  }
  if (type == "bytetrack") {
    return std::unique_ptr<ITracker>(new ByteTrackAdapter(cfg.bytetrack));
  }
  throw std::runtime_error("Unsupported tracker type: " + type);
}

}  // namespace tracking
