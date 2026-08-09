// Reads a storage buffer, doubles each value and writes it into an rgba32f image.

#include <iostream>
#include <vector>

#include "vulkan_app/vulkan_buffer.h"
#include "vulkan_app/vulkan_command_buffer.h"
#include "vulkan_app/vulkan_device.h"
#include "vulkan_app/vulkan_image.h"
#include "vulkan_app/vulkan_pipeline.h"
#include "vulkan_app/vulkan_utils.h"

#include "example_utils.h"

using namespace vulkan;

int main(int argc, char** argv) try {
    // Step 1: Initialize VulkanDevice
    VulkanDevice vulkanDevice(true); // Enable validation layers

    VkDevice device = vulkanDevice.getDevice();
    VkPhysicalDevice physicalDevice = vulkanDevice.getPhysicalDevice();
    VkCommandPool commandPool = vulkanDevice.getCommandPool();
    VkQueue queue = vulkanDevice.getComputeQueue();

    // Step 2: Sizes. The shader indexes the buffer as `y * 16 + x` over a
    // 16x16 workgroup, so the image is 16x16 and the buffer holds 256 floats.
    const uint32_t width = 16, height = 16, channels = 4;
    const size_t bufferElementCount = width * height;
    const VkDeviceSize bufferByteSize = bufferElementCount * sizeof(float);
    const size_t imageElementCount = width * height * channels;
    const VkDeviceSize imageByteSize = imageElementCount * sizeof(float);

    VulkanBuffer inputBuffer(device, physicalDevice, bufferByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VulkanImage imageOut(device, physicalDevice, width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    const std::vector<float> inputData(bufferElementCount, 1.0f);
    inputBuffer.upload(inputData.data());

    // The image must be in GENERAL layout to be used as a storage image.
    VulkanUtils::transitionImageLayout(device, commandPool, queue, imageOut.getImage(),
                                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // Step 3: Create the compute pipeline
    VulkanPipeline vulkanPipeline(device);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // inputBuffer
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },   // imageOut
    };
    vulkanPipeline.createDescriptorSetLayout(bindings);
    vulkanPipeline.createPipeline(example::shaderPath("buffer_image.spv", argc, argv), "main");

    // Step 4: Allocate the descriptor set and point it at the resources
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 }
    };
    vulkanPipeline.createDescriptorPool(poolSizes);
    VkDescriptorSet descriptorSet = vulkanPipeline.allocateDescriptorSet(vulkanPipeline.getDescriptorPool());

    VkDescriptorBufferInfo bufferInfo = { inputBuffer.getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorImageInfo imageInfo = { VK_NULL_HANDLE, imageOut.getImageView(), VK_IMAGE_LAYOUT_GENERAL };

    std::vector<VkWriteDescriptorSet> descriptorWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo, nullptr, nullptr },
    };
    vulkanPipeline.updateDescriptorSet(descriptorSet, descriptorWrites);

    // Step 5: Record and submit the dispatch. One 16x16 workgroup covers the image.
    VulkanCommandBuffer commandBuffer(device, commandPool);
    commandBuffer.beginRecording();
    commandBuffer.dispatchCompute(vulkanPipeline.getPipeline(), vulkanPipeline.getPipelineLayout(),
                                  descriptorSet, 1, 1, 1);
    commandBuffer.endRecording();
    commandBuffer.submit(queue);

    // Step 6: Read back the image and verify
    VulkanUtils::transitionImageLayout(device, commandPool, queue, imageOut.getImage(),
                                      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VulkanBuffer stagingBuffer(device, physicalDevice, imageByteSize,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanUtils::copyImageToBuffer(device, commandPool, queue, imageOut.getImage(),
                                   stagingBuffer.getBuffer(), width, height);

    std::vector<float> outputData(imageElementCount);
    stagingBuffer.download(outputData.data());

    bool success = true;
    for (size_t i = 0; i < imageElementCount; i += channels) {
        const float expectedValue = 2.0f;  // input value (1.0) * 2.0 in the shader
        if (outputData[i] != expectedValue || outputData[i + 1] != expectedValue ||
            outputData[i + 2] != expectedValue || outputData[i + 3] != expectedValue) {
            std::cerr << "Mismatch at pixel " << i / channels << ": expected (" << expectedValue
                      << "), got (" << outputData[i] << ", " << outputData[i + 1] << ", "
                      << outputData[i + 2] << ", " << outputData[i + 3] << ")" << std::endl;
            success = false;
            break;
        }
    }

    if (!success) {
        std::cerr << "Test failed: some values are incorrect." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Test passed: all values are correct." << std::endl;
    return EXIT_SUCCESS;

} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
