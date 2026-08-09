// Adds two rgba32f storage images element-wise into a third image.

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

    // Step 2: Create the images. rgba32f means 4 floats per pixel.
    const uint32_t width = 16, height = 16, channels = 4;
    const size_t elementCount = width * height * channels;
    const VkDeviceSize imageByteSize = elementCount * sizeof(float);

    VulkanImage imageA(device, physicalDevice, width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VulkanImage imageB(device, physicalDevice, width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VulkanImage imageOut(device, physicalDevice, width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Step 3: Upload the inputs through a staging buffer. Device-local images
    // cannot be mapped, so the data goes host -> buffer -> image.
    VulkanBuffer stagingBuffer(device, physicalDevice, imageByteSize,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    const std::vector<float> inputDataA(elementCount, 1.0f);
    const std::vector<float> inputDataB(elementCount, 2.0f);

    struct Upload {
        VkImage image;
        const std::vector<float>* data;
        const char* name;
    };
    const Upload uploads[] = {
        { imageA.getImage(), &inputDataA, "imageA" },
        { imageB.getImage(), &inputDataB, "imageB" },
    };

    for (const Upload& upload : uploads) {
        std::cout << "Copying data to " << upload.name << std::endl;
        stagingBuffer.upload(upload.data->data());
        VulkanUtils::transitionImageLayout(device, commandPool, queue, upload.image,
                                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VulkanUtils::copyBufferToImage(device, commandPool, queue, stagingBuffer.getBuffer(),
                                       upload.image, width, height);
        VulkanUtils::transitionImageLayout(device, commandPool, queue, upload.image,
                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    }

    VulkanUtils::transitionImageLayout(device, commandPool, queue, imageOut.getImage(),
                                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // Step 4: Create the compute pipeline
    VulkanPipeline vulkanPipeline(device);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // imageA
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // imageB
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }   // imageOut
    };
    vulkanPipeline.createDescriptorSetLayout(bindings);
    vulkanPipeline.createPipeline(example::shaderPath("images_add.spv", argc, argv), "main");

    // Step 5: Allocate the descriptor set and point it at the images
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 }
    };
    vulkanPipeline.createDescriptorPool(poolSizes);
    VkDescriptorSet descriptorSet = vulkanPipeline.allocateDescriptorSet(vulkanPipeline.getDescriptorPool());

    VkDescriptorImageInfo imageInfoA = { VK_NULL_HANDLE, imageA.getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo imageInfoB = { VK_NULL_HANDLE, imageB.getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo imageInfoOut = { VK_NULL_HANDLE, imageOut.getImageView(), VK_IMAGE_LAYOUT_GENERAL };

    std::vector<VkWriteDescriptorSet> descriptorWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfoA, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfoB, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfoOut, nullptr, nullptr }
    };
    vulkanPipeline.updateDescriptorSet(descriptorSet, descriptorWrites);

    // Step 6: Record and submit the dispatch. The shader is 16x16 per workgroup.
    VulkanCommandBuffer commandBuffer(device, commandPool);
    commandBuffer.beginRecording();
    commandBuffer.dispatchCompute(vulkanPipeline.getPipeline(), vulkanPipeline.getPipelineLayout(),
                                  descriptorSet, width / 16, height / 16, 1);
    commandBuffer.endRecording();
    commandBuffer.submit(queue);

    // Step 7: Read back imageOut and verify
    std::cout << "Copying data from imageOut" << std::endl;
    VulkanUtils::transitionImageLayout(device, commandPool, queue, imageOut.getImage(),
                                      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VulkanUtils::copyImageToBuffer(device, commandPool, queue, imageOut.getImage(),
                                   stagingBuffer.getBuffer(), width, height);

    std::vector<float> outputData(elementCount);
    stagingBuffer.download(outputData.data());

    bool success = true;
    for (size_t i = 0; i < elementCount; i += channels) {
        const float expectedValue = 3.0f;  // imageA + imageB = 1.0f + 2.0f
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
        std::cerr << "Image addition failed." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Image addition was successful!" << std::endl;
    return EXIT_SUCCESS;

} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
