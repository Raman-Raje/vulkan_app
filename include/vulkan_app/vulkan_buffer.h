#ifndef VULKAN_BUFFER_H
#define VULKAN_BUFFER_H

#include <vulkan/vulkan.h>

namespace vulkan {

class VulkanBuffer {
public:
    VulkanBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    ~VulkanBuffer();

    VkBuffer getBuffer() const;
    VkDeviceMemory getBufferMemory() const;
    VkDeviceSize getSize() const;

    // Host copies for HOST_VISIBLE | HOST_COHERENT buffers. `size` defaults to
    // the whole buffer and must not exceed it.
    void upload(const void* data, VkDeviceSize size = VK_WHOLE_SIZE);
    void download(void* data, VkDeviceSize size = VK_WHOLE_SIZE) const;

private:
    VkDevice device;
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
    VkDeviceSize size;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkPhysicalDevice physicalDevice);
    void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    void allocateMemory(VkDevice device, VkPhysicalDevice physicalDevice, VkMemoryRequirements memRequirements, VkMemoryPropertyFlags properties);
};

} // namespace vulkan

#endif // VULKAN_BUFFER_H
