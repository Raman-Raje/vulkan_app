// Verifies that VulkanTimer reports real GPU execution time.
//
// A timer that returns a constant, or ticks mislabelled as nanoseconds, or a
// stale value from a previous run, all still "look like" timings. So rather than
// asserting an absolute duration, these checks pin down the properties that only
// a correct timer has:
//
//   1. scaling      - doubling/quadrupling the GPU work scales the reported time
//                     by the same factor (catches a constant or unscaled value)
//   2. wall-clock   - reported time never exceeds the CPU-observed duration of
//                     the submission, and accounts for most of it (pins the
//                     absolute magnitude, so a wrong timestampPeriod is caught)
//   3. empty        - an empty command buffer reports a near-zero duration
//                     (catches a fixed offset baked into the result)
//   4. reuse        - the same timer used again reports the new workload rather
//                     than the previous one
//   5. spread       - repeated identical runs agree with each other
//   6. validation   - the timestamp commands themselves are spec-legal, checked
//                     by a debug messenger rather than by measurement
//
// Check 6 exists because misuse of a query pool is *undefined behaviour*, not
// reliably observable behaviour: dropping the vkCmdResetQueryPool in begin()
// still produces perfectly linear, correct-looking timings on a lenient driver
// (verified on MoltenVK/Apple M1), so no amount of measurement catches it. The
// validation layer reports it as VUID-vkCmdWriteTimestamp-None-00830.
//
// Timings are noisy, so the numeric bounds are deliberately loose: wide enough
// that normal scheduling jitter passes, tight enough that the failure modes
// above are caught.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "vulkan_app/shader_path.h"
#include "vulkan_app/vulkan_buffer.h"
#include "vulkan_app/vulkan_command_buffer.h"
#include "vulkan_app/vulkan_device.h"
#include "vulkan_app/vulkan_pipeline.h"
#include "vulkan_app/vulkan_timer.h"

using namespace vulkan;

namespace {

int failureCount = 0;

void check(bool passed, const std::string& name, const std::string& detail) {
    std::cout << (passed ? "[ ok ] " : "[FAIL] ") << name;
    if (!detail.empty()) {
        std::cout << " -- " << detail;
    }
    std::cout << std::endl;
    if (!passed) {
        ++failureCount;
    }
}

void skip(const std::string& name, const std::string& reason) {
    std::cout << "[skip] " << name << " -- " << reason << std::endl;
}

std::string describe(const std::string& label, double value, double low, double high) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << label << " = " << value << ", expected " << low
        << " .. " << high;
    return out.str();
}

// Collects validation-layer errors so the test can fail on them. VulkanDevice
// enables VK_EXT_debug_utils together with the validation layers, so the
// messenger can be attached to its instance from here.
class ValidationMonitor {
public:
    explicit ValidationMonitor(VkInstance instance) : instance_(instance) {
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (create == nullptr) {
            return;  // validation layers / debug utils not enabled
        }

        VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        createInfo.pfnUserCallback = &ValidationMonitor::onMessage;
        createInfo.pUserData = this;

        if (create(instance_, &createInfo, nullptr, &messenger_) != VK_SUCCESS) {
            messenger_ = VK_NULL_HANDLE;
        }
    }

    ~ValidationMonitor() {
        if (messenger_ == VK_NULL_HANDLE) {
            return;
        }
        auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy != nullptr) {
            destroy(instance_, messenger_, nullptr);
        }
    }

    ValidationMonitor(const ValidationMonitor&) = delete;
    ValidationMonitor& operator=(const ValidationMonitor&) = delete;

    bool attached() const { return messenger_ != VK_NULL_HANDLE; }
    const std::vector<std::string>& messages() const { return messages_; }

private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL onMessage(
        VkDebugUtilsMessageSeverityFlagBitsEXT /*severity*/,
        VkDebugUtilsMessageTypeFlagsEXT /*types*/,
        const VkDebugUtilsMessengerCallbackDataEXT* data, void* userData) {
        auto* self = static_cast<ValidationMonitor*>(userData);
        if (self != nullptr && data != nullptr && data->pMessage != nullptr) {
            self->messages_.emplace_back(data->pMessageIdName != nullptr ? data->pMessageIdName
                                                                        : "(unnamed)");
        }
        return VK_FALSE;
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    std::vector<std::string> messages_;
};

struct Measurement {
    double gpuMillis = 0.0;
    double wallMillis = 0.0;
};

// Owns everything needed to submit the tunable workload repeatedly.
class WorkloadRunner {
public:
    explicit WorkloadRunner(bool enableValidation)
        : vulkanDevice_(enableValidation),
          device_(vulkanDevice_.getDevice()),
          physicalDevice_(vulkanDevice_.getPhysicalDevice()),
          queue_(vulkanDevice_.getComputeQueue()),
          commandPool_(vulkanDevice_.getCommandPool()),
          monitor_(vulkanDevice_.getInstance()),
          paramsBuffer_(device_, physicalDevice_, sizeof(uint32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
          dataBuffer_(device_, physicalDevice_, sizeof(float) * kElementCount,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
          pipeline_(device_) {
        const std::vector<float> initialData(kElementCount, 1.0f);
        dataBuffer_.upload(initialData.data());

        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // params
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // data
        };
        pipeline_.createDescriptorSetLayout(bindings);
        pipeline_.createPipeline(shaderPath("timer_workload.spv"), "main");

        std::vector<VkDescriptorPoolSize> poolSizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 }
        };
        pipeline_.createDescriptorPool(poolSizes);
        descriptorSet_ = pipeline_.allocateDescriptorSet(pipeline_.getDescriptorPool());

        VkDescriptorBufferInfo paramsInfo = { paramsBuffer_.getBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dataInfo = { dataBuffer_.getBuffer(), 0, VK_WHOLE_SIZE };
        std::vector<VkWriteDescriptorSet> writes = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &paramsInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dataInfo, nullptr },
        };
        pipeline_.updateDescriptorSet(descriptorSet_, writes);
    }

    // Runs the workload once, timed by both `timer` and the CPU clock.
    Measurement run(VulkanTimer& timer, uint32_t iterations) {
        paramsBuffer_.upload(&iterations, sizeof(iterations));

        VulkanCommandBuffer commandBuffer(device_, commandPool_);
        commandBuffer.beginRecording();
        timer.begin(commandBuffer.getCommandBuffer());
        commandBuffer.dispatchCompute(pipeline_.getPipeline(), pipeline_.getPipelineLayout(),
                                      descriptorSet_, kElementCount / 256, 1, 1);
        timer.end(commandBuffer.getCommandBuffer());
        commandBuffer.endRecording();

        // submit() ends in vkQueueWaitIdle, so this interval contains the whole
        // GPU execution and the timestamps are readable once it returns.
        const auto wallStart = std::chrono::steady_clock::now();
        commandBuffer.submit(queue_);
        const auto wallEnd = std::chrono::steady_clock::now();

        Measurement measurement;
        measurement.gpuMillis = timer.getElapsedMillis();
        measurement.wallMillis =
            std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();
        return measurement;
    }

    // Records timestamps around a command buffer that does no work at all.
    double runEmpty(VulkanTimer& timer) {
        VulkanCommandBuffer commandBuffer(device_, commandPool_);
        commandBuffer.beginRecording();
        timer.begin(commandBuffer.getCommandBuffer());
        timer.end(commandBuffer.getCommandBuffer());
        commandBuffer.endRecording();
        commandBuffer.submit(queue_);
        return timer.getElapsedMillis();
    }

    VkDevice device() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    uint32_t queueFamilyIndex() const { return vulkanDevice_.getComputeQueueFamilyIndex(); }
    const ValidationMonitor& monitor() const { return monitor_; }

private:
    static const uint32_t kElementCount = 4096;

    VulkanDevice vulkanDevice_;
    VkDevice device_;
    VkPhysicalDevice physicalDevice_;
    VkQueue queue_;
    VkCommandPool commandPool_;
    ValidationMonitor monitor_;
    VulkanBuffer paramsBuffer_;
    VulkanBuffer dataBuffer_;
    VulkanPipeline pipeline_;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
};

// Iteration count of the baseline workload. Chosen so the baseline runs for a
// few milliseconds: long enough that timer resolution and submission overhead
// do not dominate the measurement.
const uint32_t kBaseIterations = 200000;

// Median of repeated runs, so a single scheduling hiccup cannot fail the
// scaling checks.
double medianGpuMillis(WorkloadRunner& runner, VulkanTimer& timer, uint32_t iterations, int runs) {
    std::vector<double> samples;
    samples.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        samples.push_back(runner.run(timer, iterations).gpuMillis);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Validation layers are wanted here (check 6), but must not be a hard
// requirement: fall back to running without them.
std::unique_ptr<WorkloadRunner> makeRunner(bool& validationEnabled) {
    try {
        auto runner = std::make_unique<WorkloadRunner>(true);
        validationEnabled = true;
        return runner;
    } catch (const std::exception& e) {
        std::cout << "note: could not enable validation layers (" << e.what()
                  << "); continuing without them" << std::endl;
    }
    validationEnabled = false;
    return std::make_unique<WorkloadRunner>(false);
}

} // namespace

int main() try {
    bool validationEnabled = false;
    std::unique_ptr<WorkloadRunner> runnerPtr = makeRunner(validationEnabled);
    WorkloadRunner& runner = *runnerPtr;

    VulkanTimer timer(runner.device(), runner.physicalDevice(), runner.queueFamilyIndex());

    // Warm up: the first dispatch pays for shader/pipeline warm-up on some
    // drivers, which would distort the baseline.
    runner.run(timer, kBaseIterations);

    std::cout << "--- baseline (" << kBaseIterations << " iterations) ---" << std::endl;
    const Measurement baseline = runner.run(timer, kBaseIterations);
    std::cout << std::fixed << std::setprecision(3) << "gpu = " << baseline.gpuMillis
              << " ms, wall = " << baseline.wallMillis << " ms" << std::endl;

    // 0. The measurement must be positive; everything below builds on it.
    check(baseline.gpuMillis > 0.0, "reports a positive duration for real work",
          describe("gpu ms", baseline.gpuMillis, 0.0, 1e9));

    // A baseline near zero means the workload is too small for the ratio checks
    // below to carry any signal.
    if (baseline.gpuMillis < 0.05) {
        std::cout << "\nbaseline too short to verify scaling reliably; "
                  << "raise kBaseIterations for this GPU." << std::endl;
        return failureCount == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    // 1. Scaling: the core property. A timer that ignores the workload, or
    //    scales it by the wrong factor, cannot track these ratios.
    std::cout << "\n--- scaling ---" << std::endl;
    const double base = medianGpuMillis(runner, timer, kBaseIterations, 5);
    const double doubled = medianGpuMillis(runner, timer, kBaseIterations * 2, 5);
    const double quadrupled = medianGpuMillis(runner, timer, kBaseIterations * 4, 5);

    const double doubleRatio = doubled / base;
    const double quadrupleRatio = quadrupled / base;
    std::cout << "1x = " << base << " ms, 2x = " << doubled << " ms (" << doubleRatio << "x), "
              << "4x = " << quadrupled << " ms (" << quadrupleRatio << "x)" << std::endl;

    check(doubleRatio > 1.6 && doubleRatio < 2.5, "doubling the work doubles the reported time",
          describe("ratio", doubleRatio, 1.6, 2.5));
    check(quadrupleRatio > 3.2 && quadrupleRatio < 5.0,
          "quadrupling the work quadruples the reported time",
          describe("ratio", quadrupleRatio, 3.2, 5.0));

    // 2. Cross-check against the CPU clock. GPU execution happens strictly
    //    inside the submit interval, so it cannot exceed it; and for a workload
    //    this long it should account for most of it. This is what pins down the
    //    absolute scale, i.e. that timestampPeriod is applied correctly.
    std::cout << "\n--- cpu cross-check ---" << std::endl;
    const Measurement crossCheck = runner.run(timer, kBaseIterations * 4);
    const double fraction = crossCheck.gpuMillis / crossCheck.wallMillis;
    std::cout << "gpu = " << crossCheck.gpuMillis << " ms, wall = " << crossCheck.wallMillis
              << " ms (gpu is " << (fraction * 100.0) << "% of wall)" << std::endl;

    check(crossCheck.gpuMillis <= crossCheck.wallMillis * 1.05,
          "reported time does not exceed the CPU-observed duration",
          describe("gpu/wall", fraction, 0.0, 1.05));
    check(fraction > 0.5, "reported time accounts for most of the submission",
          describe("gpu/wall", fraction, 0.5, 1.05));

    // 3. An empty command buffer: top-of-pipe to bottom-of-pipe with nothing in
    //    between. A sizeable value here means a fixed offset is leaking in.
    std::cout << "\n--- empty command buffer ---" << std::endl;
    const double emptyMillis = runner.runEmpty(timer);
    std::cout << "gpu = " << emptyMillis << " ms" << std::endl;
    check(emptyMillis >= 0.0 && emptyMillis < 0.5, "empty command buffer reports near-zero",
          describe("gpu ms", emptyMillis, 0.0, 0.5));

    // 4. Reuse: a second use of the same timer must report the new workload.
    //    Note this is a weak check on lenient drivers, which overwrite
    //    timestamps even without a reset; check 6 is what covers that case.
    std::cout << "\n--- reuse after a longer run ---" << std::endl;
    runner.run(timer, kBaseIterations * 4);
    const double afterLong = runner.run(timer, kBaseIterations).gpuMillis;
    const double staleRatio = afterLong / base;
    std::cout << "long run then 1x = " << afterLong << " ms (" << staleRatio
              << "x the 1x baseline)" << std::endl;
    check(staleRatio > 0.5 && staleRatio < 2.0,
          "a reused timer reports the new run, not the previous one",
          describe("ratio vs 1x", staleRatio, 0.5, 2.0));

    // 5. Repeatability: identical submissions should agree. Wild variation would
    //    suggest results are being read before the work completes.
    std::cout << "\n--- repeatability (8 identical runs) ---" << std::endl;
    std::vector<double> samples;
    for (int i = 0; i < 8; ++i) {
        samples.push_back(runner.run(timer, kBaseIterations).gpuMillis);
    }
    std::sort(samples.begin(), samples.end());
    const double spread = samples.back() / samples.front();
    std::cout << "min = " << samples.front() << " ms, median = " << samples[samples.size() / 2]
              << " ms, max = " << samples.back() << " ms (spread " << spread << "x)" << std::endl;
    check(spread < 3.0, "identical runs report consistent times",
          describe("max/min", spread, 1.0, 3.0));

    // 6. Spec compliance of the query-pool usage itself.
    std::cout << "\n--- validation layer ---" << std::endl;
    const std::string validationCheck = "no validation errors from the timestamp commands";
    if (!validationEnabled) {
        skip(validationCheck, "validation layers not available");
    } else if (!runner.monitor().attached()) {
        skip(validationCheck, "could not attach a debug messenger");
    } else {
        const std::vector<std::string>& messages = runner.monitor().messages();
        std::ostringstream detail;
        detail << messages.size() << " message(s)";
        for (size_t i = 0; i < messages.size() && i < 5; ++i) {
            detail << (i == 0 ? ": " : ", ") << messages[i];
        }
        check(messages.empty(), validationCheck, detail.str());
    }

    std::cout << "\n" << (failureCount == 0 ? "all timer checks passed" : "timer checks FAILED")
              << " (" << failureCount << " failure(s))" << std::endl;
    return failureCount == 0 ? EXIT_SUCCESS : EXIT_FAILURE;

} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
