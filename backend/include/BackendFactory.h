#pragma once

#include "IBackend.h"
#include <memory>
#include <string>

namespace backend {

class BackendFactory {
public:
    static std::shared_ptr<IBackend> create(const BackendConfig& cfg);
};

} // namespace backend
