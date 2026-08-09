#ifndef VULKAN_TIMER_H
#define VULKAN_TIMER_H

#include <vulkan/vulkan.h>
#include <cstdint>

namespace vulkan {

// Measures GPU execution time using Vulkan timestamp queries.
//
// begin() and end() only *record* the timestamp writes, so they must be called
// between beginRecording() and endRecording() of the command buffer that runs
// the work being measured. The timestamps are readable once that command buffer
// has been submitted and has completed:
//
//     VulkanTimer timer(device, physicalDevice, vulkanDevice.getComputeQueueFamilyIndex());
//     commandBuffer.beginRecording();
//     timer.begin(commandBuffer.getCommandBuffer());
//     commandBuffer.dispatchCompute(...);
//     timer.end(commandBuffer.getCommandBuffer());
//     commandBuffer.endRecording();
//     commandBuffer.submit(queue);
//     std::cout << timer.getElapsedMillis() << " ms\n";
class VulkanTimer {
public:
    VulkanTimer(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex);
    ~VulkanTimer();

    // Resets the query pool and writes the start timestamp (top of pipe).
    void begin(VkCommandBuffer commandBuffer);

    // Writes the end timestamp (bottom of pipe).
    void end(VkCommandBuffer commandBuffer);

    // Waits for the timestamps and returns the elapsed GPU time.
    int64_t getElapsedNanos() const;
    double getElapsedMillis() const;

private:
    VkDevice device;
    VkQueryPool queryPool;
    float timestampPeriod;   // nanoseconds per timestamp tick
    uint64_t timestampMask;  // meaningful bits of a timestamp on this queue family

    static const uint32_t kStartQuery = 0;
    static const uint32_t kEndQuery = 1;
    static const uint32_t kQueryCount = 2;

    void createQueryPool();
};

} // namespace vulkan

#endif // VULKAN_TIMER_H
