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

    // Define input tensor shape (1, 8, 2, 2) for testing layout transformation
    const int n = 1, c = 4, h = 16, w = 16;
    VkDeviceSize imageSize = w * h * c * sizeof(float);  // Assuming RGBA32F format

    // Create input images (imageA, imageB)
    VulkanImage imageA(device, physicalDevice, w, h, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VulkanImage imageB(device, physicalDevice, w, h, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Create output image (imageOut)
    VulkanImage imageOut(device, physicalDevice, w, h, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);    

    // Fill imageA and imageB with data
    std::vector<float> inputDataA(imageSize, 1.0f); // Fill with 1.0f
    std::vector<float> inputDataB(imageSize, 2.0f); // Fill with 2.0f

    // ++++++++++++++++++++ copy data to image. ++++++++++++++++++++++++
    // Step 1: Create a staging buffer
    VulkanBuffer stagingBuffer(device, physicalDevice, imageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    std::cout<<"Copying data to imageA"<<std::endl;

    // Step 2: Map the staging buffer and copy data
    void* data;
    vkMapMemory(device, stagingBuffer.getBufferMemory(), 0, imageSize, 0, &data);
    std::memcpy(data, inputDataA.data(), static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBuffer.getBufferMemory());

    // Transition imageA to TRANSFER_DST_OPTIMAL for copying data to it
    VulkanUtils::transitionImageLayout(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(), imageA.getImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VulkanUtils::copyBufferToImage(device,vulkanDevice.getCommandPool(),vulkanDevice.getComputeQueue(),stagingBuffer.getBuffer(),imageA.getImage(),w,h);
    // After copying, transition imageA to GENERAL for shader read/write access
    VulkanUtils::transitionImageLayout(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(), imageA.getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    

    std::cout<<"Copying data to imageB"<<std::endl;

    // For image B
    vkMapMemory(device, stagingBuffer.getBufferMemory(), 0, imageSize, 0, &data);
    std::memcpy(data, inputDataB.data(), static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBuffer.getBufferMemory());

    // Transition imageA to TRANSFER_DST_OPTIMAL for copying data to it
    VulkanUtils::transitionImageLayout(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(), imageB.getImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VulkanUtils::copyBufferToImage(device,vulkanDevice.getCommandPool(),vulkanDevice.getComputeQueue(),stagingBuffer.getBuffer(),imageB.getImage(),w,h);
    // After copying, transition imageA to GENERAL for shader read/write access
    VulkanUtils::transitionImageLayout(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(), imageB.getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);


    VulkanUtils::transitionImageLayout(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(), imageOut.getImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // pipeline creation
    VulkanPipeline vulkanPipeline(device);

    // Define Descriptor Set Layout Bindings for bufferA, bufferB, and bufferOut
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // bufferA
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // bufferB
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }   // bufferOut
    };
    vulkanPipeline.createDescriptorSetLayout(bindings);

    // Create pipeline with the buffer addition shader
    vulkanPipeline.createPipeline("/local/mnt/workspace/ramashin/tests/vulkan/testSpirv/spirv/images_add.spv", "main");

    //Allocate Descriptor Set and Update with Resources
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 }  // 3 buffers (bufferA, bufferB, bufferOut)
    };
    vulkanPipeline.createDescriptorPool(poolSizes);

    VkDescriptorSet descriptorSet = vulkanPipeline.allocateDescriptorSet(vulkanPipeline.getDescriptorPool());

    // Create descriptor image info structs
    VkDescriptorImageInfo imageInfoA = {};
    imageInfoA.imageView = imageA.getImageView();
    imageInfoA.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo imageInfoB = {};
    imageInfoB.imageView = imageB.getImageView();
    imageInfoB.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo imageInfoOut = {};
    imageInfoOut.imageView = imageOut.getImageView();
    imageInfoOut.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::vector<VkWriteDescriptorSet> descriptorWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfoA, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfoB, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfoOut, nullptr, nullptr }
    };

    vulkanPipeline.updateDescriptorSet(descriptorSet, descriptorWrites);

    // Record Command Buffer and Dispatch Compute Shader
    VulkanCommandBuffer commandBuffer(device, vulkanDevice.getCommandPool());
    commandBuffer.beginRecording();

    // Dispatch the compute shader
    commandBuffer.dispatchCompute(vulkanPipeline.getPipeline(), vulkanPipeline.getPipelineLayout(), descriptorSet, w / 16, h / 16, 1);
    commandBuffer.endRecording();

    // Submit the command buffer
    commandBuffer.submit(vulkanDevice.getComputeQueue());

    // Step 7: Verify the Results
    std::cout<<"Copying data from imageOut"<<std::endl;

    // Transition the output image layout to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL for copying to buffer
    VulkanUtils::transitionImageLayout(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(), imageOut.getImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Copy the image data to the staging buffer
    VulkanUtils::copyImageToBuffer(device, vulkanDevice.getCommandPool(), vulkanDevice.getComputeQueue(), imageOut.getImage(), stagingBuffer.getBuffer(), w, h);

    // Map the staging buffer and read back the transformed data
    float* mappedData;
    vkMapMemory(device, stagingBuffer.getBufferMemory(), 0, imageSize, 0, (void**)&mappedData);

    // Print or verify the transformed data
    bool success = true;
    for (int i = 0; i < w * h * c; i += 4) {
        float expectedValue = 3.0f; // imageA + imageB = 1.0f + 2.0f = 3.0f
        if (mappedData[i] != expectedValue || mappedData[i + 1] != expectedValue || mappedData[i + 2] != expectedValue || mappedData[i + 3] != expectedValue) {
            std::cerr << "Mismatch at pixel index " << i / 4 << ": expected (" << expectedValue << "), got ("
                      << mappedData[i] << ", " << mappedData[i + 1] << ", " << mappedData[i + 2] << ", " << mappedData[i + 3] << ")" << std::endl;
            success = false;
        }
    }
    vkUnmapMemory(device, stagingBuffer.getBufferMemory());

    if (success) {
        std::cout << "Image addition was successful!" << std::endl;
    } else {
        std::cerr << "Image addition failed." << std::endl;
    }


    return 0;
}