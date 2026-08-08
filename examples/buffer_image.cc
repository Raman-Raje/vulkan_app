#include <cstring>
#include <iostream>
#include <vector>
#include <vulkan/vulkan_core.h>
#include "vulkan_buffer.h"
#include "vulkan_image.h"
#include "vulkan_device.h"
#include "vulkan_utils.h"
#include "vulkan_pipeline.h"
#include "vulkan_command_buffer.h"

using namespace vulkan;

int main() {

    // Step 1: Initialize VulkanDevice
    VulkanDevice vulkanDevice(true); // Enable validation layers

    // Step 2: Create Buffers
    VkDevice device = vulkanDevice.getDevice();
    VkPhysicalDevice physicalDevice = vulkanDevice.getPhysicalDevice();

    // image size

    const int height = 4, width = 4, channel = 4, length = 1024;
    VkDeviceSize imageSize = height * width * channel * sizeof(float);
    VkDeviceSize bufferSize = length * sizeof(float);

    // create input buffer
    VulkanBuffer inputBuffer(device,physicalDevice, bufferSize,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); 

    // create output image
    VulkanImage imageOut(device, physicalDevice, width, height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);    


    // Fill buffer with data
    std::vector<float> inputDataB(bufferSize, 1.0f); // Fill with 1.0f

    // Map the data to buffer
    void* mappedData;
    vkMapMemory(device, inputBuffer.getBufferMemory(), 0, imageSize, 0, &mappedData);
    std::memcpy(mappedData, inputDataB.data(), static_cast<size_t>(bufferSize));
    vkUnmapMemory(device, inputBuffer.getBufferMemory());


    // Pipeline creation
    VulkanPipeline vulkanPipeline(device);

    // Define Descriptor Set Layout Bindings for bufferA, bufferB, and bufferOut
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, 
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // bufferB
    };
    vulkanPipeline.createDescriptorSetLayout(bindings); 

    // Create pipeline with the buffer addition shader
    vulkanPipeline.createPipeline("/local/mnt/workspace/ramashin/tests/vulkan/testSpirv/spirv/buffer_image.spv", "main");

    //Allocate Descriptor Set and Update with Resources
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },  // 3 buffers (bufferA, bufferB, bufferOut)
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 }  // 3 buffers (bufferA, bufferB, bufferOut)
    };
    vulkanPipeline.createDescriptorPool(poolSizes);

  VkDescriptorSet descriptorSet = vulkanPipeline.allocateDescriptorSet(vulkanPipeline.getDescriptorPool());

    // Create descriptor buffer info struct
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = inputBuffer.getBuffer();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    // Create descriptor image info struct
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.imageView = imageOut.getImageView();
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::vector<VkWriteDescriptorSet> descriptorWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo, nullptr, nullptr },
    };

    vulkanPipeline.updateDescriptorSet(descriptorSet, descriptorWrites);

    // Step 5: Record Command Buffer and Dispatch Compute Shader
    VulkanCommandBuffer commandBuffer(device, vulkanDevice.getCommandPool());
    commandBuffer.beginRecording();

    commandBuffer.beginRecording();

    // Transition the output image to GENERAL layout for read/write access in the shader
    VulkanUtils::transitionImageLayout(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(),
                                       imageOut.getImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
                                       
    // Dispatch the compute shader
    commandBuffer.dispatchCompute(vulkanPipeline.getPipeline(), vulkanPipeline.getPipelineLayout(), descriptorSet, bufferSize / 256, 1, 1);  // Adjust workgroup size as needed

    commandBuffer.endRecording();

    // Submit the command buffer
    commandBuffer.submit(vulkanDevice.getComputeQueue());

    // Step 6: Verify the Results
    // Transition the output image layout to TRANSFER_SRC_OPTIMAL for copying to buffer
    VulkanUtils::transitionImageLayout(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(),
                                       imageOut.getImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Create a staging buffer to copy the image data back to the CPU
    VulkanBuffer stagingBuffer(device, physicalDevice, imageSize,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Copy the image data to the staging buffer
    VulkanUtils::copyImageToBuffer(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(),
                                   imageOut.getImage(), stagingBuffer.getBuffer(), width, height);

    // Map the staging buffer and read back the data
    vkMapMemory(device, stagingBuffer.getBufferMemory(), 0, imageSize, 0, &mappedData);

    // Print or verify the transformed data
    bool success = true;
    float* outputData = static_cast<float*>(mappedData);
    for (int i = 0; i < width * height * channel; i += 4) {
        float expectedValue = 2.0f; // input value (1.0) * 2.0 from shader
        if (outputData[i] != expectedValue || outputData[i + 1] != expectedValue ||
            outputData[i + 2] != expectedValue || outputData[i + 3] != expectedValue) {
            std::cerr << "Mismatch at index " << i / 4 << ": expected (" << expectedValue << "), got ("
                      << outputData[i] << ", " << outputData[i + 1] << ", " << outputData[i + 2] << ", "
                      << outputData[i + 3] << ")" << std::endl;
            success = false;
        }
    }

    vkUnmapMemory(device, stagingBuffer.getBufferMemory());

    if (success) {
        std::cout << "Test passed: all values are correct." << std::endl;
    } else {
        std::cout << "Test failed: some values are incorrect." << std::endl;
    }     

    return 0;

}