#ifndef VULKAN_PIPELINE_H
#define VULKAN_PIPELINE_H

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

namespace vulkan {

class VulkanPipeline {
public:
    VulkanPipeline(VkDevice device);
    ~VulkanPipeline();

    void createPipeline(const std::string& shaderPath,const std::string& entry_point);
    VkPipeline getPipeline() const;
    VkPipelineLayout getPipelineLayout() const;

    // Descriptor management
    void createDescriptorSetLayout(const std::vector<VkDescriptorSetLayoutBinding>& bindings);
    VkDescriptorSetLayout getDescriptorSetLayout() const;

    // Descriptor pool
    void createDescriptorPool(const std::vector<VkDescriptorPoolSize>& pools);
    VkDescriptorPool getDescriptorPool() const;    

    // Descriptor set allocation and update
    VkDescriptorSet allocateDescriptorSet(VkDescriptorPool descriptorPool);
    void updateDescriptorSet(VkDescriptorSet descriptorSet, const std::vector<VkWriteDescriptorSet>& descriptorWrites);

private:
    VkDevice device;
    VkPipeline pipeline;
    VkShaderModule shaderModule;
    VkPipelineLayout pipelineLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout descriptorSetLayout;

    void createShaderModule(const std::vector<char>& code);
    std::vector<char> readFile(const std::string& filename);
};

} // namespace vulkan

#endif // VULKAN_PIPELINE_H
