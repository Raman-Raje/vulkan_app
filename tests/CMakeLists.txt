add_executable(timer_test timer_test.cc)
target_link_libraries(timer_test PRIVATE vulkan_app)

add_test(NAME timer_test COMMAND timer_test)
set_tests_properties(timer_test PROPERTIES
  # The test submits ~30 timed workloads; generous timeout for slow/loaded GPUs.
  TIMEOUT 300
  # Backstop for validation output that reaches stderr instead of the test's own
  # debug messenger (e.g. a platform where the messenger cannot be attached).
  FAIL_REGULAR_EXPRESSION "Validation Error"
)
