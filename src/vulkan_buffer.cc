#include "vulkan_app/vulkan_buffer.h"

#include <cstring>
#include <stdexcept>

namespace vulkan {

VulkanBuffer::VulkanBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
    : device(device), buffer(VK_NULL_HANDLE), bufferMemory(VK_NULL_HANDLE), size(size) {
    createBuffer(device, physicalDevice, size, usage, properties);
}

VulkanBuffer::~VulkanBuffer() {
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
    }
    if (bufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, bufferMemory, nullptr);
    }
}

uint32_t VulkanBuffer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanBuffer::createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    allocateMemory(device, physicalDevice, memRequirements, properties);
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void VulkanBuffer::allocateMemory(VkDevice device, VkPhysicalDevice physicalDevice, VkMemoryRequirements memRequirements, VkMemoryPropertyFlags properties) {
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, physicalDevice);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }
}

VkBuffer VulkanBuffer::getBuffer() const {
    return buffer;
}

VkDeviceMemory VulkanBuffer::getBufferMemory() const {
    return bufferMemory;
}

VkDeviceSize VulkanBuffer::getSize() const {
    return size;
}

void VulkanBuffer::upload(const void* data, VkDeviceSize copySize) {
    if (copySize == VK_WHOLE_SIZE) {
        copySize = size;
    }
    if (copySize > size) {
        throw std::runtime_error("upload size exceeds buffer size!");
    }

    void* mapped;
    if (vkMapMemory(device, bufferMemory, 0, copySize, 0, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("failed to map buffer memory!");
    }
    std::memcpy(mapped, data, static_cast<size_t>(copySize));
    vkUnmapMemory(device, bufferMemory);
}

void VulkanBuffer::download(void* data, VkDeviceSize copySize) const {
    if (copySize == VK_WHOLE_SIZE) {
        copySize = size;
    }
    if (copySize > size) {
        throw std::runtime_error("download size exceeds buffer size!");
    }

    void* mapped;
    if (vkMapMemory(device, bufferMemory, 0, copySize, 0, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("failed to map buffer memory!");
    }
    std::memcpy(data, mapped, static_cast<size_t>(copySize));
    vkUnmapMemory(device, bufferMemory);
}

} // namespace vulkan
