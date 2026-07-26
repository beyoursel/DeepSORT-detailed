#pragma once

#include "IDetector.h"
#include <memory>
#include <string>

namespace detector {

class DetectorFactory {
public:
    static std::shared_ptr<IDetector> create(const std::string& type);
};

} // namespace detector
