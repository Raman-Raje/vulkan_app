cmake_minimum_required(VERSION 3.16)
project(vulkan_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(VULKAN_APP_BUILD_EXAMPLES "Build the example programs" ON)

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
# on the command line.
target_compile_definitions(vulkan_app PUBLIC
  VULKAN_APP_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/spv"
)

if(VULKAN_APP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

install(TARGETS vulkan_app ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
