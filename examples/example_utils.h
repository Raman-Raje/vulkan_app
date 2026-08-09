#ifndef VULKAN_APP_EXAMPLE_UTILS_H
#define VULKAN_APP_EXAMPLE_UTILS_H

#include <string>

#include "vulkan_app/shader_path.h"

namespace example {

// Resolves the SPIR-V module for an example: argv[1] when the caller passes one,
// otherwise `name` inside the configured shader directory.
inline std::string shaderPath(const std::string& name, int argc, char** argv) {
    if (argc > 1) {
        return std::string(argv[1]);
    }
    return vulkan::shaderPath(name);
}

} // namespace example

#endif // VULKAN_APP_EXAMPLE_UTILS_H
