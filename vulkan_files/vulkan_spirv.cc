#include <cstring>
#include <iostream>
#include <vector>
#include "vulkan_buffer.h"
#include "vulkan_image.h"
#include "vulkan_device.h"
#include "vulkan_utils.h"
#include "vulkan_pipeline.h"
#include "vulkan_command_buffer.h"

using namespace vulkan;

int main() {


try {
    // Step 1: Initialize VulkanDevice
    VulkanDevice vulkanDevice(true); // Enable validation layers

    // Step 2: Create Buffers and Images
    VkDevice device = vulkanDevice.getDevice();
    VkPhysicalDevice physicalDevice = vulkanDevice.getPhysicalDevice();

    // Define input tensor shape (1, 8, 2, 2) for testing layout transformation
    const int n = 1, c = 8, h = 2, w = 2;
    VkDeviceSize bufferSize = sizeof(float) * n * c * h * w; // 32 floats

    // Sample input tensor data in (n, c, h, w) format
    std::vector<float> inputTensor = {
        // c = 0
        1.0f, 2.0f, 3.0f, 4.0f,
        // c = 1
        5.0f, 6.0f, 7.0f, 8.0f,
        // c = 2
        9.0f, 10.0f, 11.0f, 12.0f,
        // c = 3
        13.0f, 14.0f, 15.0f, 16.0f,
        // c = 4
        17.0f, 18.0f, 19.0f, 20.0f,
        // c = 5
        21.0f, 22.0f, 23.0f, 24.0f,
        // c = 6
        25.0f, 26.0f, 27.0f, 28.0f,
        // c = 7
        29.0f, 30.0f, 31.0f, 32.0f
    };

    // Create a buffer to act as the input
    VulkanBuffer inputBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Copy input data to the Vulkan buffer
    void* data;
    vkMapMemory(device, inputBuffer.getBufferMemory(), 0, bufferSize, 0, &data);
    memcpy(data, inputTensor.data(), static_cast<size_t>(bufferSize));
    vkUnmapMemory(device, inputBuffer.getBufferMemory());

    // Create an image to act as the output (for the transformed data)
    VulkanImage outputImage(device, physicalDevice, 2, 2, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Step 3: Define Descriptor Set Layout Bindings
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },   // Image binding
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }  // Buffer binding
    };

    // Step 4: Create VulkanPipeline
    VulkanPipeline vulkanPipeline(device);
    vulkanPipeline.createDescriptorSetLayout(bindings);

    // Step 5: Create Pipeline
    vulkanPipeline.createPipeline("/local/mnt/workspace/ramashin/tests/dumps/tvmgen_default_fused_layout_transform_kernel0_spv.spv");

    // Step 6: Allocate Descriptor Set and Update with Resources
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 }
    };
    vulkanPipeline.createDescriptorPool(poolSizes);
    VkDescriptorSet descriptorSet = vulkanPipeline.allocateDescriptorSet(vulkanPipeline.getDescriptorPool());

    VkDescriptorImageInfo imageInfo = {};
    imageInfo.imageView = outputImage.getImageView();
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = inputBuffer.getBuffer();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    std::vector<VkWriteDescriptorSet> descriptorWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfo, nullptr }
    };

    vulkanPipeline.updateDescriptorSet(descriptorSet, descriptorWrites);

    // Step 7: Record Command Buffer and Dispatch Compute Shader
    VulkanCommandBuffer commandBuffer(device, vulkanDevice.getCommandPool());
    commandBuffer.beginRecording();
    commandBuffer.dispatchCompute(vulkanPipeline.getPipeline(), vulkanPipeline.getPipelineLayout(), descriptorSet);
    commandBuffer.endRecording();

    // Submit the command buffer
    commandBuffer.submit(vulkanDevice.getComputeQueue());

    // Step 8: Retrieve and Verify the Output

    // Transition the output image layout to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL for copying to buffer
    VulkanUtils::transitionImageLayout(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(), outputImage.getImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Create a staging buffer to copy the output image data back to the CPU
    VulkanBuffer stagingBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Copy the image data to the staging buffer
    VulkanUtils::copyImageToBuffer(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(), outputImage.getImage(), stagingBuffer.getBuffer(), 2, 2);

    // Map the staging buffer and read back the transformed data
    float* mappedData;
    vkMapMemory(device, stagingBuffer.getBufferMemory(), 0, bufferSize, 0, (void**)&mappedData);

    // Print or verify the transformed data
    std::cout << "Transformed Data (n, c/4, h, w, c%4 format):\n";
    for (size_t i = 0; i < 32; i += 4) {
        std::cout << mappedData[i] << ", " << mappedData[i+1] << ", " << mappedData[i+2] << ", " << mappedData[i+3] << std::endl;
    }

    vkUnmapMemory(device, stagingBuffer.getBufferMemory());

    return 0;

} catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    // Print more details or exit
    return EXIT_FAILURE;
}    

}
