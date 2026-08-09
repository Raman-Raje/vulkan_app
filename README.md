# vulkan_app

A small standalone harness for running Vulkan compute shaders, with wrapper
classes modelled on TVM's Vulkan runtime structure.

## Layout

```
include/vulkan_app/   Public headers for the wrapper library
src/                  Library implementation
examples/             One standalone main() per example
tests/                Correctness tests (run via ctest)
shaders/              GLSL compute sources
shaders/spv/          Compiled SPIR-V modules the examples load
shaders/disasm/       SPIR-V / GLSL disassembly dumps (reference only)
tools/                Shader compilation helper
```

The library is one CMake target, `vulkan_app`; each example links against it.

## Building

Requires CMake 3.16+, a C++17 compiler, and the Vulkan SDK.

```sh
cmake -S . -B build
cmake --build build -j
```

Binaries land in `build/examples/`. Configure with
`-DVULKAN_APP_BUILD_EXAMPLES=OFF` / `-DVULKAN_APP_BUILD_TESTS=OFF` to build only
the library.

## Tests

```sh
ctest --test-dir build --output-on-failure
```

`tests/timer_test` verifies `VulkanTimer` against a compute workload whose cost
is proportional to an iteration count supplied at runtime
(`shaders/timer_workload.comp`). Absolute durations are not asserted — they are
hardware-specific and noisy. Instead it checks the properties only a correct
timer has:

| Check | Catches |
| --- | --- |
| Doubling / quadrupling the work scales the reported time by the same factor | Constant or unscaled values |
| Reported time never exceeds, but accounts for most of, the CPU-observed submit duration | A wrong `timestampPeriod`, i.e. wrong absolute scale |
| An empty command buffer reports near-zero | A fixed offset baked into the result |
| A reused timer reports the new run | Stale results from a previous submission |
| Repeated identical runs agree | Results read before the work completes |
| A debug messenger sees no validation errors | Spec-illegal query-pool usage |

The last check exists because query-pool misuse is *undefined behaviour* rather
than reliably observable behaviour. Removing the `vkCmdResetQueryPool` from
`VulkanTimer::begin()` still yields perfectly linear, correct-looking timings on
MoltenVK/Apple M1 — no measurement catches it, but the validation layer reports
`VUID-vkCmdWriteTimestamp-None-00830`. The test attaches its own messenger and
fails on any validation message; validation layers are used when available and
skipped (not required) when not.

Each check was verified to have teeth by deliberately breaking the timer:

| Injected bug | Result |
| --- | --- |
| Return a constant duration | 4 checks fail |
| Drop `vkCmdResetQueryPool` | Validation check fails |
| `timestampPeriod` off by 50x | 2 checks fail |

Reference numbers on an Apple M1: scaling ratios of 2.00x and 3.98x, GPU time
96.8% of wall clock, 0.017 ms for an empty command buffer, and 2.3% spread across
eight identical runs.

## Running

Each example loads its SPIR-V from `shaders/spv/` (baked in at configure time as
`VULKAN_APP_SHADER_DIR`), so it can be run from anywhere:

```sh
./build/examples/buffer_add
```

Pass a path to override the module, e.g. to run a different TVM kernel dump
through the same host code:

```sh
./build/examples/layout_transform /path/to/kernel.spv
```

| Example | What it does |
| --- | --- |
| `buffer_add` | Element-wise add of two storage buffers; also times the dispatch |
| `image_add` | Element-wise add of two `rgba32f` storage images |
| `buffer_image` | Reads a storage buffer, doubles it, writes an `rgba32f` image |
| `layout_transform` | Runs a TVM-generated NCHW -> NCHW4c kernel (buffer -> image) |

## Classes

| Class | Responsibility |
| --- | --- |
| `VulkanDevice` | Instance, physical/logical device, compute queue, command pool |
| `VulkanBuffer` | Buffer + backing memory, with host `upload()` / `download()` |
| `VulkanImage` | Image + memory + image view |
| `VulkanPipeline` | Shader module, descriptor set layout/pool, compute pipeline |
| `VulkanCommandBuffer` | Recording, compute dispatch, submission |
| `VulkanTimer` | GPU-side timing via timestamp queries |
| `VulkanUtils` | Single-time commands, buffer/image copies, layout transitions |

### Timing a dispatch

`VulkanTimer` writes timestamps into the same command buffer as the work being
measured, so `begin()` / `end()` must be called while recording:

```cpp
VulkanTimer timer(device, physicalDevice, vulkanDevice.getComputeQueueFamilyIndex());

commandBuffer.beginRecording();
timer.begin(commandBuffer.getCommandBuffer());
commandBuffer.dispatchCompute(pipeline, layout, descriptorSet, groupCount, 1, 1);
timer.end(commandBuffer.getCommandBuffer());
commandBuffer.endRecording();
commandBuffer.submit(queue);

std::cout << timer.getElapsedMillis() << " ms\n";
```

The elapsed time is only meaningful once the command buffer has completed;
`VulkanCommandBuffer::submit()` waits for the queue to go idle, so the value is
ready as soon as it returns.

## Shaders

`shaders/spv/` is checked in, so no shader compilation is needed to run the
examples. To rebuild the hand-written shaders after editing them:

```sh
./tools/compile_shaders.sh
```

`tvmgen_default_fused_layout_transform_kernel0_spv.spv` comes from TVM;
the matching `.comp` is a disassembly for reference, not a compilable source.

`timer_workload.comp` exists only for `tests/timer_test`; its loop bound comes
from a buffer and its arithmetic forms a dependent chain, so the cost cannot be
unrolled, folded, or optimized away.

## Known issues

`layout_transform` fails on strict drivers (e.g. MoltenVK on macOS). The
TVM-generated SPIR-V declares its output image as
`OpTypeImage %float 2D 0 0 0 0 Rgba32f` — `Sampled = 0`, which the Vulkan spec
disallows (`VUID-StandaloneSpirv-OpTypeImage-04657` requires 1 or 2). Lenient
drivers accept it as a storage image; MoltenVK translates it to a sampled Metal
texture and the pipeline fails to compile. The fix belongs in how the module is
generated (it should be a storage image, `Sampled = 2`), not in the host code.
