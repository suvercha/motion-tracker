#pragma once

#include <chrono>

namespace motion {

// Monotonic time, so speeds are not disturbed by system clock changes.
using Clock = std::chrono::steady_clock;

}  // namespace motion
