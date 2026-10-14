cmake_minimum_required(VERSION 3.16)
project(vulkan_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(VULKAN_APP_BUILD_EXAMPLES "Build the example programs" ON)
option(VULKAN_APP_BUILD_TESTS "Build the tests" ON)

# Cross-compiling with the NDK toolchain file
# (-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake) sets
# ANDROID to true; this option defaults from that but can be forced
# independently. It adjusts two things that differ from a desktop build:
# the Khronos validation layer isn't present on-device by default, and the
# shader directory can't be the host build path, since nothing is deployed
# there.
option(VULKAN_APP_ANDROID "Build for Android: validation layers off by default, VULKAN_APP_SHADER_DIR points at an on-device path" ${ANDROID})

find_package(Vulkan REQUIRED)

# Core library: thin RAII wrappers over the Vulkan compute objects.
add_library(vulkan_app
  src/vulkan_device.cc
  src/vulkan_buffer.cc
  src/vulkan_image.cc
  src/vulkan_pipeline.cc
  src/vulkan_command_buffer.cc
  src/vulkan_timer.cc
  src/vulkan_utils.cc
)
add_library(vulkan_app::vulkan_app ALIAS vulkan_app)

target_include_directories(vulkan_app PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(vulkan_app PUBLIC Vulkan::Vulkan)

# Examples locate their SPIR-V through this directory unless a path is passed
# on the command line. On Android there is no host filesystem to point at;
# shaders/spv/ must be pushed to the device separately (see README) and this
# define points at wherever that landed.
if(VULKAN_APP_ANDROID)
  set(VULKAN_APP_ANDROID_SHADER_DIR "/data/local/tmp/vulkan_app/shaders/spv" CACHE STRING
      "On-device directory shaders/spv/ is pushed to, e.g. via adb push")
  target_compile_definitions(vulkan_app PUBLIC
    VULKAN_APP_SHADER_DIR="${VULKAN_APP_ANDROID_SHADER_DIR}"
    VULKAN_APP_ANDROID=1
  )
else()
  target_compile_definitions(vulkan_app PUBLIC
    VULKAN_APP_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/spv"
  )
endif()

if(VULKAN_APP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(VULKAN_APP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

install(TARGETS vulkan_app ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
