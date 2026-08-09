#ifndef VULKAN_SHADER_PATH_H
#define VULKAN_SHADER_PATH_H

#include <string>

// Set by CMake to <repo>/shaders/spv. The fallback keeps this header usable
// when a translation unit is compiled by hand from the repository root.
#ifndef VULKAN_APP_SHADER_DIR
#define VULKAN_APP_SHADER_DIR "shaders/spv"
#endif

namespace vulkan {

// Directory the checked-in SPIR-V modules live in.
inline std::string shaderDir() {
    return VULKAN_APP_SHADER_DIR;
}

// Absolute path of a checked-in SPIR-V module, e.g. shaderPath("buffer_add.spv").
inline std::string shaderPath(const std::string& name) {
    return shaderDir() + "/" + name;
}

} // namespace vulkan

#endif // VULKAN_SHADER_PATH_H
