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

    // Step 1: Initialize VulkanDevice
    VulkanDevice vulkanDevice(true); // Enable validation layers

    // Step 2: Create Buffers
    VkDevice device = vulkanDevice.getDevice();
    VkPhysicalDevice physicalDevice = vulkanDevice.getPhysicalDevice();

    const int bufferSize = 1024; // Number of elements
    VkDeviceSize bufferByteSize = sizeof(float) * bufferSize;

    // Create input buffers (bufferA, bufferB)
    VulkanBuffer bufferA(device, physicalDevice, bufferByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer bufferB(device, physicalDevice, bufferByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Create output buffer (bufferOut)
    VulkanBuffer bufferOut(device, physicalDevice, bufferByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Fill bufferA and bufferB with data
    std::vector<float> inputDataA(bufferSize, 1.0f); // Fill with 1.0f
    std::vector<float> inputDataB(bufferSize, 2.0f); // Fill with 2.0f

    // Map and copy data to bufferA
    void* data;
    vkMapMemory(device, bufferA.getBufferMemory(), 0, bufferByteSize, 0, &data);
    std::memcpy(data, inputDataA.data(), bufferByteSize);
    vkUnmapMemory(device, bufferA.getBufferMemory());

    // Map and copy data to bufferB
    vkMapMemory(device, bufferB.getBufferMemory(), 0, bufferByteSize, 0, &data);
    std::memcpy(data, inputDataB.data(), bufferByteSize);
    vkUnmapMemory(device, bufferB.getBufferMemory());

    // Step 3: Create Vulkan Pipeline
    // Assume buffer_add.spv is the SPIR-V binary compiled from the GLSL shader for adding buffers
    VulkanPipeline vulkanPipeline(device);

    // Define Descriptor Set Layout Bindings for bufferA, bufferB, and bufferOut
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // bufferA
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // bufferB
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }   // bufferOut
    };
    vulkanPipeline.createDescriptorSetLayout(bindings);

    // Create pipeline with the buffer addition shader
    vulkanPipeline.createPipeline("/local/mnt/workspace/ramashin/tests/vulkan/testSpirv/spirv/buffer_add.spv","main");

    // Step 4: Allocate Descriptor Set and Update with Resources
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 }  // 3 buffers (bufferA, bufferB, bufferOut)
    };
    vulkanPipeline.createDescriptorPool(poolSizes);
    VkDescriptorSet descriptorSet = vulkanPipeline.allocateDescriptorSet(vulkanPipeline.getDescriptorPool());

    // Create descriptor buffer info structs
    VkDescriptorBufferInfo bufferInfoA = {};
    bufferInfoA.buffer = bufferA.getBuffer();
    bufferInfoA.offset = 0;
    bufferInfoA.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo bufferInfoB = {};
    bufferInfoB.buffer = bufferB.getBuffer();
    bufferInfoB.offset = 0;
    bufferInfoB.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo bufferInfoOut = {};
    bufferInfoOut.buffer = bufferOut.getBuffer();
    bufferInfoOut.offset = 0;
    bufferInfoOut.range = VK_WHOLE_SIZE;

    std::vector<VkWriteDescriptorSet> descriptorWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfoA, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfoB, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfoOut, nullptr }
    };

    vulkanPipeline.updateDescriptorSet(descriptorSet, descriptorWrites);

    // Step 5: Record Command Buffer and Dispatch Compute Shader
    VulkanCommandBuffer commandBuffer(device, vulkanDevice.getCommandPool());
    commandBuffer.beginRecording();

    // Dispatch the compute shader
    commandBuffer.dispatchCompute(vulkanPipeline.getPipeline(), vulkanPipeline.getPipelineLayout(), descriptorSet, bufferSize / 256, 1, 1);  // Adjust workgroup size as needed

    commandBuffer.endRecording();

    // Submit the command buffer
    commandBuffer.submit(vulkanDevice.getComputeQueue());

    // Step 6: Verify the Results
    // Map and read back data from bufferOut
    float* outputData;
    vkMapMemory(device, bufferOut.getBufferMemory(), 0, bufferByteSize, 0, (void**)&outputData);

    bool success = true;
    for (int i = 0; i < bufferSize; ++i) {
        if (outputData[i] != 3.0f) {  // bufferA + bufferB = 1.0f + 2.0f = 3.0f
            std::cerr << "Mismatch at index " << i << ": expected 3.0f, got " << outputData[i] << std::endl;
            success = false;
        }
    }
    vkUnmapMemory(device, bufferOut.getBufferMemory());

    if (success) {
        std::cout << "Buffer addition was successful!" << std::endl;
    } else {
        std::cerr << "Buffer addition failed." << std::endl;
    }

    return 0;
}