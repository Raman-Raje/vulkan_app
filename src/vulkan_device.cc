#include "vulkan_app/vulkan_device.h"
#include <iostream>
#include <string.h>
#include <vector>
#include <set>

namespace vulkan {

VulkanDevice::VulkanDevice(bool enableValidationLayers, bool enableCooperativeMatrix)
    : instance(VK_NULL_HANDLE), physicalDevice(VK_NULL_HANDLE), device(VK_NULL_HANDLE), computeQueue(VK_NULL_HANDLE), commandPool(VK_NULL_HANDLE), computeQueueFamilyIndex(0), enableValidationLayers(enableValidationLayers), enableCooperativeMatrix(enableCooperativeMatrix) {
    createInstance();
    pickPhysicalDevice();
    createLogicalDevice();
    commandPool = createCommandPool(); // Initialize the command pool during construction
}

VulkanDevice::~VulkanDevice() {
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
    }
    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
}

void VulkanDevice::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Compute Application";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    // VK_KHR_cooperative_matrix depends on features promoted into core 1.1-1.3
    // (16-bit storage, shaderFloat16, subgroup-size-control); core 1.0 doesn't
    // have those feature structs at all.
    appInfo.apiVersion = enableCooperativeMatrix ? VK_API_VERSION_1_3 : VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

#ifdef __APPLE__
    // MoltenVK is a portability driver, so the loader only enumerates it when
    // the instance opts in.
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create Vulkan instance!");
    }
}

void VulkanDevice::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice = device;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

bool VulkanDevice::isDeviceSuitable(VkPhysicalDevice device) {
    uint32_t queueFamilyIndex = findQueueFamilies(device);
    return queueFamilyIndex != UINT32_MAX;

}

uint32_t VulkanDevice::findQueueFamilies(VkPhysicalDevice device) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamilyIndex = i;
            return i;
        }
    }

    return UINT32_MAX; // No suitable queue family found
}

void VulkanDevice::createLogicalDevice() {
    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures = {};

    // Step 1: Specify the required device extensions
    std::vector<const char*> requiredExtensions = { VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME };
    if (enableCooperativeMatrix) {
        requiredExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }

    // Step 2: Check if the device supports the required extensions
    if (!checkDeviceExtensionSupport(physicalDevice, requiredExtensions)) {
        throw std::runtime_error("required device extensions are not supported!");
    }

    // A portability driver (e.g. MoltenVK) requires this extension to be enabled
    // whenever it advertises it. Spelled out rather than using the header macro,
    // which is only visible with VK_ENABLE_BETA_EXTENSIONS.
    const char* kPortabilitySubset = "VK_KHR_portability_subset";
    if (checkDeviceExtensionSupport(physicalDevice, { kPortabilitySubset })) {
        requiredExtensions.push_back(kPortabilitySubset);
    }

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
    createInfo.ppEnabledExtensionNames = requiredExtensions.data();

    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    // Step 3: For cooperative matrix, negotiate the feature chain it depends
    // on (16-bit storage, shaderFloat16, subgroup-size-control) through
    // VkPhysicalDeviceFeatures2 rather than the plain VkPhysicalDeviceFeatures
    // struct, and verify every bit we're about to request is actually
    // supported before asking vkCreateDevice to enable it.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeatures = {};
    coopMatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;

    VkPhysicalDeviceVulkan13Features vulkan13Features = {};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &coopMatFeatures;

    VkPhysicalDeviceVulkan12Features vulkan12Features = {};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.pNext = &vulkan13Features;

    VkPhysicalDeviceVulkan11Features vulkan11Features = {};
    vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan11Features.pNext = &vulkan12Features;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &vulkan11Features;

    if (enableCooperativeMatrix) {
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

        if (!vulkan11Features.storageBuffer16BitAccess) {
            throw std::runtime_error("device does not support storageBuffer16BitAccess!");
        }
        if (!vulkan12Features.shaderFloat16) {
            throw std::runtime_error("device does not support shaderFloat16!");
        }
        if (!vulkan13Features.subgroupSizeControl || !vulkan13Features.computeFullSubgroups) {
            throw std::runtime_error("device does not support subgroupSizeControl/computeFullSubgroups!");
        }
        if (!coopMatFeatures.cooperativeMatrix) {
            throw std::runtime_error("device does not support VK_KHR_cooperative_matrix (cooperativeMatrix feature)!");
        }

        // Re-enable exactly the features we checked for (vkGetPhysicalDeviceFeatures2
        // may have reported other, unrelated features/extensions as supported too).
        vulkan11Features.storageBuffer16BitAccess = VK_TRUE;
        vulkan12Features.shaderFloat16 = VK_TRUE;
        vulkan13Features.subgroupSizeControl = VK_TRUE;
        vulkan13Features.computeFullSubgroups = VK_TRUE;
        coopMatFeatures.cooperativeMatrix = VK_TRUE;

        createInfo.pNext = &features2;
        createInfo.pEnabledFeatures = nullptr;
    } else {
        createInfo.pEnabledFeatures = &deviceFeatures;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
}

VkDevice VulkanDevice::getDevice() const {
    return device;
}

VkPhysicalDevice VulkanDevice::getPhysicalDevice() const {
    return physicalDevice;
}

VkInstance VulkanDevice::getInstance() const {
    return instance;
}

VkQueue VulkanDevice::getComputeQueue() const {
    return computeQueue;
}

uint32_t VulkanDevice::getComputeQueueFamilyIndex() const {
    return computeQueueFamilyIndex;
}

VkCommandPool VulkanDevice::getCommandPool() const {
    return commandPool;
}

uint32_t VulkanDevice::getSubgroupSize() const {
    VkPhysicalDeviceSubgroupProperties subgroupProperties = {};
    subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &subgroupProperties;

    vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);
    return subgroupProperties.subgroupSize;
}

std::vector<VkCooperativeMatrixPropertiesKHR> VulkanDevice::getCooperativeMatrixProperties() const {
    auto vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (!vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
        throw std::runtime_error("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR not available "
                                  "(VK_KHR_cooperative_matrix not enabled)!");
    }

    uint32_t count = 0;
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &count, nullptr);

    std::vector<VkCooperativeMatrixPropertiesKHR> properties(count);
    for (auto& prop : properties) {
        prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    }
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &count, properties.data());
    return properties;
}

VkCommandPool VulkanDevice::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    poolInfo.flags = 0; // Optional

    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    return commandPool;
}

bool VulkanDevice::checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

std::vector<const char*> VulkanDevice::getRequiredExtensions() {
    std::vector<const char*> extensions;

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    // Required by the VK_KHR_portability_subset device extension that
    // createLogicalDevice() enables on portability drivers.
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

    return extensions;
}

bool VulkanDevice::checkDeviceExtensionSupport(VkPhysicalDevice device, const std::vector<const char*>& requiredExtensions) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensionsSet(requiredExtensions.begin(), requiredExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensionsSet.erase(extension.extensionName);
    }

    return requiredExtensionsSet.empty();
}


}
