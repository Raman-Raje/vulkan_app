#ifndef VULKAN_DEVICE_H
#define VULKAN_DEVICE_H

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

namespace vulkan {

// The Khronos validation layer isn't present on Android devices by default
// (unlike a desktop Vulkan SDK install), so VulkanDevice(true) there throws
// "validation layers requested, but not available!" before anything runs.
// Examples pass this instead of a hardcoded `true` so the same source works
// on both.
#ifdef VULKAN_APP_ANDROID
constexpr bool kDefaultEnableValidationLayers = false;
#else
constexpr bool kDefaultEnableValidationLayers = true;
#endif

class VulkanDevice {
public:
    // enableCooperativeMatrix requests VK_KHR_cooperative_matrix plus the
    // float16 / 16-bit-storage / subgroup-size-control features it depends
    // on, bumping the instance/device to Vulkan 1.3. Throws if the physical
    // device doesn't support them.
    VulkanDevice(bool enableValidationLayers, bool enableCooperativeMatrix = false);
    ~VulkanDevice();

    VkDevice getDevice() const;
    VkPhysicalDevice getPhysicalDevice() const;
    VkInstance getInstance() const;
    VkQueue getComputeQueue() const;
    uint32_t getComputeQueueFamilyIndex() const;
    VkCommandPool createCommandPool();
    VkCommandPool getCommandPool() const;

    // Subgroup size the device actually runs compute shaders with. Needed to
    // size `local_size_x_id = 0` specialization constants that assume one
    // workgroup == one subgroup (cooperative-matrix shaders).
    uint32_t getSubgroupSize() const;

    // Cooperative-matrix (M,N,K,type) configurations this device supports.
    // Only meaningful when constructed with enableCooperativeMatrix = true.
    std::vector<VkCooperativeMatrixPropertiesKHR> getCooperativeMatrixProperties() const;

private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;

    bool enableValidationLayers;
    bool enableCooperativeMatrix;

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
