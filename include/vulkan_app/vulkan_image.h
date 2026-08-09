#ifndef VULKAN_IMAGE_H
#define VULKAN_IMAGE_H

#include <vulkan/vulkan.h>

namespace vulkan {

class VulkanImage {
public:
    VulkanImage(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags properties);
    ~VulkanImage();

    VkImage getImage() const;
    VkImageView getImageView() const;
    VkDeviceMemory getImageMemory() const;

private:
    VkDevice device;
    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView imageView;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkPhysicalDevice physicalDevice);
    void createImage(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags properties);
    void createImageView(VkDevice device, VkFormat format);
    void allocateMemory(VkDevice device, VkPhysicalDevice physicalDevice, VkMemoryRequirements memRequirements, VkMemoryPropertyFlags properties);
};

} // namespace vulkan

#endif // VULKAN_IMAGE_H
