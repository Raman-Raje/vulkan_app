#include "vulkan_app/vulkan_timer.h"

#include <stdexcept>
#include <vector>

namespace vulkan {

VulkanTimer::VulkanTimer(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex)
    : device(device), queryPool(VK_NULL_HANDLE), timestampPeriod(0.0f), timestampMask(0) {
    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    timestampPeriod = properties.limits.timestampPeriod;

    // Only the low `timestampValidBits` of a timestamp are meaningful, and a
    // queue family that reports zero cannot write timestamps at all.
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    if (queueFamilyIndex >= queueFamilyCount) {
        throw std::runtime_error("invalid queue family index for timestamp queries!");
    }

    uint32_t validBits = queueFamilies[queueFamilyIndex].timestampValidBits;
    if (validBits == 0) {
        throw std::runtime_error("queue family does not support timestamp queries!");
    }
    timestampMask = (validBits >= 64) ? ~static_cast<uint64_t>(0)
                                     : ((static_cast<uint64_t>(1) << validBits) - 1);

    createQueryPool();
}

VulkanTimer::~VulkanTimer() {
    if (queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, queryPool, nullptr);
    }
}

void VulkanTimer::createQueryPool() {
    VkQueryPoolCreateInfo queryPoolInfo = {};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = kQueryCount;

    if (vkCreateQueryPool(device, &queryPoolInfo, nullptr, &queryPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create query pool!");
    }
}

void VulkanTimer::begin(VkCommandBuffer commandBuffer) {
    // Queries must be reset before they are written to.
    vkCmdResetQueryPool(commandBuffer, queryPool, kStartQuery, kQueryCount);
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, kStartQuery);
}

void VulkanTimer::end(VkCommandBuffer commandBuffer) {
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, kEndQuery);
}

int64_t VulkanTimer::getElapsedNanos() const {
    uint64_t timestamps[kQueryCount] = {0};

    VkResult result = vkGetQueryPoolResults(device, queryPool, kStartQuery, kQueryCount,
                                            sizeof(timestamps), timestamps, sizeof(uint64_t),
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to get query pool results!");
    }

    uint64_t elapsedTicks = (timestamps[kEndQuery] & timestampMask) -
                            (timestamps[kStartQuery] & timestampMask);
    return static_cast<int64_t>(elapsedTicks * timestampPeriod);
}

double VulkanTimer::getElapsedMillis() const {
    return static_cast<double>(getElapsedNanos()) / 1e6;
}

} // namespace vulkan
