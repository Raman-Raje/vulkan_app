// Adds two storage buffers element-wise and times the dispatch on the GPU.

#include <iostream>
#include <vector>

#include "vulkan_app/vulkan_buffer.h"
#include "vulkan_app/vulkan_command_buffer.h"
#include "vulkan_app/vulkan_device.h"
#include "vulkan_app/vulkan_pipeline.h"
#include "vulkan_app/vulkan_timer.h"

#include "example_utils.h"

using namespace vulkan;

int main(int argc, char** argv) try {
    // Step 1: Initialize VulkanDevice
    VulkanDevice vulkanDevice(true); // Enable validation layers

    VkDevice device = vulkanDevice.getDevice();
    VkPhysicalDevice physicalDevice = vulkanDevice.getPhysicalDevice();

    // Step 2: Create Buffers
    const int elementCount = 1024;
    const VkDeviceSize bufferByteSize = sizeof(float) * elementCount;

    const VkMemoryPropertyFlags hostVisible =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VulkanBuffer bufferA(device, physicalDevice, bufferByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostVisible);
    VulkanBuffer bufferB(device, physicalDevice, bufferByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostVisible);
    VulkanBuffer bufferOut(device, physicalDevice, bufferByteSize,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hostVisible);

    std::vector<float> inputDataA(elementCount, 1.0f);
    std::vector<float> inputDataB(elementCount, 2.0f);
    bufferA.upload(inputDataA.data());
    bufferB.upload(inputDataB.data());

    // Step 3: Create the compute pipeline
    VulkanPipeline vulkanPipeline(device);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // bufferA
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // bufferB
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }   // bufferOut
    };
    vulkanPipeline.createDescriptorSetLayout(bindings);
    vulkanPipeline.createPipeline(example::shaderPath("buffer_add.spv", argc, argv), "main");

    // Step 4: Allocate the descriptor set and point it at the buffers
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 }
    };
    vulkanPipeline.createDescriptorPool(poolSizes);
    VkDescriptorSet descriptorSet = vulkanPipeline.allocateDescriptorSet(vulkanPipeline.getDescriptorPool());

    VkDescriptorBufferInfo bufferInfoA = { bufferA.getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo bufferInfoB = { bufferB.getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo bufferInfoOut = { bufferOut.getBuffer(), 0, VK_WHOLE_SIZE };

    std::vector<VkWriteDescriptorSet> descriptorWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfoA, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfoB, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfoOut, nullptr }
    };
    vulkanPipeline.updateDescriptorSet(descriptorSet, descriptorWrites);

    // Step 5: Record the dispatch, wrapped in timestamp queries
    VulkanTimer timer(device, physicalDevice, vulkanDevice.getComputeQueueFamilyIndex());
    VulkanCommandBuffer commandBuffer(device, vulkanDevice.getCommandPool());

    commandBuffer.beginRecording();
    timer.begin(commandBuffer.getCommandBuffer());
    // The shader has local_size_x = 256, so one workgroup covers 256 elements.
    commandBuffer.dispatchCompute(vulkanPipeline.getPipeline(), vulkanPipeline.getPipelineLayout(),
                                  descriptorSet, elementCount / 256, 1, 1);
    timer.end(commandBuffer.getCommandBuffer());
    commandBuffer.endRecording();

    // submit() waits for the queue to go idle, so the timestamps are ready.
    commandBuffer.submit(vulkanDevice.getComputeQueue());

    std::cout << "GPU time: " << timer.getElapsedMillis() << " ms ("
              << timer.getElapsedNanos() << " ns)" << std::endl;

    // Step 6: Verify the results
    std::vector<float> outputData(elementCount);
    bufferOut.download(outputData.data());

    bool success = true;
    for (int i = 0; i < elementCount; ++i) {
        if (outputData[i] != 3.0f) {  // bufferA + bufferB = 1.0f + 2.0f
            std::cerr << "Mismatch at index " << i << ": expected 3, got " << outputData[i] << std::endl;
            success = false;
            break;
        }
    }

    if (!success) {
        std::cerr << "Buffer addition failed." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Buffer addition was successful!" << std::endl;
    return EXIT_SUCCESS;

} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
