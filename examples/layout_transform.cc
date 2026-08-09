// Runs a TVM-generated layout transform kernel: a (1, 4, 5, 5) NCHW buffer is
// written out as a 5x5 rgba32f image (NCHW4c), i.e. the channel dimension is
// packed into the image's four components.

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

    // Step 2: Sizes. The kernel runs 25 invocations, each gathering 4 values
    // (strided by 25) from the buffer into one pixel of a 5x5 image.
    const uint32_t width = 5, height = 5, channels = 4;
    const size_t bufferElementCount = width * height * channels;  // 100 floats
    const VkDeviceSize bufferByteSize = bufferElementCount * sizeof(float);
    const VkDeviceSize imageByteSize = width * height * channels * sizeof(float);

    VulkanBuffer inputBuffer(device, physicalDevice, bufferByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VulkanImage imageOut(device, physicalDevice, width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    std::vector<float> inputData(bufferElementCount);
    for (size_t i = 0; i < bufferElementCount; ++i) {
        inputData[i] = static_cast<float>(i);  // 0, 1, 2, ..., 99
    }
    inputBuffer.upload(inputData.data());

    // The image must be in GENERAL layout to be used as a storage image.
    VulkanUtils::transitionImageLayout(device, commandPool, queue, imageOut.getImage(),
                                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // Step 3: Create the compute pipeline. Note the TVM entry point name.
    VulkanPipeline vulkanPipeline(device);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },   // imageOut
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // inputBuffer
    };
    vulkanPipeline.createDescriptorSetLayout(bindings);
    vulkanPipeline.createPipeline(
        example::shaderPath("tvmgen_default_fused_layout_transform_kernel0_spv.spv", argc, argv),
        "tvmgen_default_fused_layout_transform_kernel0");

    // Step 4: Allocate the descriptor set and point it at the resources
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 }
    };
    vulkanPipeline.createDescriptorPool(poolSizes);
    VkDescriptorSet descriptorSet = vulkanPipeline.allocateDescriptorSet(vulkanPipeline.getDescriptorPool());

    VkDescriptorImageInfo imageInfo = { VK_NULL_HANDLE, imageOut.getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo bufferInfo = { inputBuffer.getBuffer(), 0, VK_WHOLE_SIZE };

    std::vector<VkWriteDescriptorSet> descriptorWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfo, nullptr },
    };
    vulkanPipeline.updateDescriptorSet(descriptorSet, descriptorWrites);

    // Step 5: Record and submit the dispatch. The kernel's local size is 25, so
    // a single workgroup covers the whole 5x5 image.
    VulkanCommandBuffer commandBuffer(device, commandPool);
    commandBuffer.beginRecording();
    commandBuffer.dispatchCompute(vulkanPipeline.getPipeline(), vulkanPipeline.getPipelineLayout(),
                                  descriptorSet, 1, 1, 1);
    commandBuffer.endRecording();
    commandBuffer.submit(queue);

    // Step 6: Read back the image
    VulkanUtils::transitionImageLayout(device, commandPool, queue, imageOut.getImage(),
                                      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VulkanBuffer stagingBuffer(device, physicalDevice, imageByteSize,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanUtils::copyImageToBuffer(device, commandPool, queue, imageOut.getImage(),
                                   stagingBuffer.getBuffer(), width, height);

    std::vector<float> outputData(width * height * channels);
    stagingBuffer.download(outputData.data());

    // Pixel (x, y) is expected to hold buffer elements i, i+25, i+50, i+75
    // where i = y * 5 + x.
    bool success = true;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t pixel = y * width + x;
            const float* actual = &outputData[pixel * channels];

            std::cout << "Pixel " << pixel << ": (" << actual[0] << ", " << actual[1] << ", "
                      << actual[2] << ", " << actual[3] << ")" << std::endl;

            for (uint32_t c = 0; c < channels; ++c) {
                const float expected = static_cast<float>(pixel + c * width * height);
                if (actual[c] != expected) {
                    std::cerr << "  mismatch in component " << c << ": expected " << expected
                              << ", got " << actual[c] << std::endl;
                    success = false;
                }
            }
        }
    }

    if (!success) {
        std::cerr << "Layout transform failed." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Layout transform was successful!" << std::endl;
    return EXIT_SUCCESS;

} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
