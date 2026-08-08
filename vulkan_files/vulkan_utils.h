#ifndef VULKAN_UTILS_H
#define VULKAN_UTILS_H

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

namespace vulkan {

class VulkanUtils {
public:
    // Command buffer management
    static VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool);
    static void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkCommandBuffer commandBuffer);

    // Buffer-to-image and image-to-buffer copy operations
    // static void  copyDataToImage(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkImage image, void* data, uint32_t width, uint32_t height);
    static void copyBufferToImage(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    static void copyImageToBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkImage image, VkBuffer buffer, uint32_t width, uint32_t height);

    // Image layout transitions
    static void transitionImageLayout(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);

    // Shader module creation
    static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);

    // File reading helper
    static std::vector<char> readFile(const std::string& filename);
};

} // namespace vulkan

#endif // VULKAN_UTILS_H
