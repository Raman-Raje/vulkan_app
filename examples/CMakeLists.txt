# Each example is a standalone main() linked against the core library.
set(VULKAN_APP_EXAMPLES
  buffer_add
  image_add
  buffer_image
  layout_transform
)

foreach(example IN LISTS VULKAN_APP_EXAMPLES)
  add_executable(${example} ${example}.cc)
  target_link_libraries(${example} PRIVATE vulkan_app)
endforeach()
