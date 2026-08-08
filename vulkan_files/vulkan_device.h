#ifndef VULKAN_DEVICE_H
#define VULKAN_DEVICE_H

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

namespace vulkan {

class VulkanDevice {
public:
    VulkanDevice(bool enableValidationLayers);
    ~VulkanDevice();

    VkDevice getDevice() const;
    VkPhysicalDevice getPhysicalDevice() const;
    VkInstance getInstance() const;
    VkQueue getComputeQueue() const;
    VkCommandPool createCommandPool();
    VkCommandPool getCommandPool() const;

private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;

    bool enableValidationLayers;

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    void createInstance();
    void pickPhysicalDevice();
    bool isDeviceSuitable(VkPhysicalDevice device);
    uint32_t findQueueFamilies(VkPhysicalDevice device);
    void createLogicalDevice();

    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    bool checkDeviceExtensionSupport(VkPhysicalDevice device, const std::vector<const char*>& requiredExtensions);
};

} // namespace vulkan

#endif // VULKAN_DEVICE_H
