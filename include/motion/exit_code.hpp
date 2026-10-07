#pragma once

namespace motion {

// Process exit codes shared by the programs.
enum class ExitCode { Ok = 0, NotFound = 1, Error = 2 };

[[nodiscard]] constexpr int toInt(ExitCode code) { return static_cast<int>(code); }

}  // namespace motion
