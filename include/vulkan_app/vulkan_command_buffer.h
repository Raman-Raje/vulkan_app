#ifndef VULKAN_COMMAND_BUFFER_H
#define VULKAN_COMMAND_BUFFER_H

#include <vulkan/vulkan.h>
#include <vector>

namespace vulkan {

class VulkanCommandBuffer {
public:
    VulkanCommandBuffer(VkDevice device, VkCommandPool commandPool);
    ~VulkanCommandBuffer();

    void beginRecording();
    void endRecording();
    void submit(VkQueue queue, VkFence fence = VK_NULL_HANDLE);
    void dispatchCompute(VkPipeline pipeline, VkPipelineLayout pipelineLayout, VkDescriptorSet descriptorSet, uint32_t groupCountX = 1, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);

    VkCommandBuffer getCommandBuffer() const;

private:
    VkDevice device;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
};

} // namespace vulkan

#endif // VULKAN_COMMAND_BUFFER_H
