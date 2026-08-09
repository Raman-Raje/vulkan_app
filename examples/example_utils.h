#ifndef VULKAN_APP_EXAMPLE_UTILS_H
#define VULKAN_APP_EXAMPLE_UTILS_H

#include <string>

// Set by CMake to <repo>/shaders/spv; the fallback keeps the header usable when
// an example is compiled by hand from the repository root.
#ifndef VULKAN_APP_SHADER_DIR
#define VULKAN_APP_SHADER_DIR "shaders/spv"
#endif

namespace example {

// Resolves the SPIR-V module for an example: argv[1] when the caller passes one,
// otherwise `name` inside the configured shader directory.
inline std::string shaderPath(const std::string& name, int argc, char** argv) {
    if (argc > 1) {
        return std::string(argv[1]);
    }
    return std::string(VULKAN_APP_SHADER_DIR) + "/" + name;
}

} // namespace example

#endif // VULKAN_APP_EXAMPLE_UTILS_H
