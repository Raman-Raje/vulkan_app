// Runs the same fp16 GEMM (C = A * B) through two cooperative-matrix shaders
// that differ only in memory layout, verifies both against a CPU reference,
// and compares their dispatch time:
//
//   gemm_kfirst        B stored transposed ("K-first"); tile loads stride by
//                       the full K dimension.
//   gemm_kfirst_tiled   A and B additionally re-packed into TILE_K-wide
//                       contiguous strips; tile loads stride by TILE_K only.
//
// Requires a device that supports VK_KHR_cooperative_matrix with a float16
// subgroup-scope configuration (recent NVIDIA/AMD/Intel/ARM GPUs). This is
// not the case on software/CPU Vulkan implementations such as llvmpipe.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "vulkan_app/shader_path.h"
#include "vulkan_app/vulkan_buffer.h"
#include "vulkan_app/vulkan_command_buffer.h"
#include "vulkan_app/vulkan_device.h"
#include "vulkan_app/vulkan_pipeline.h"
#include "vulkan_app/vulkan_timer.h"

using namespace vulkan;

namespace {

// --- Minimal IEEE-754 binary16 <-> binary32 conversion -------------------
// Scoped to this example: the shaders exchange float16_t buffers, and the
// host has nothing better than float32 to generate/verify data with. Treats
// subnormal halves as zero and does not handle NaN/Inf specially beyond
// passing the sign/exponent pattern through; the GEMM data generated below
// never reaches those ranges.

uint16_t floatToHalf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent <= 0) {
        return static_cast<uint16_t>(sign);  // too small for a normal half -> flush to zero
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> infinity
    }

    uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
    if (mantissa & 0x1000u) {
        half = static_cast<uint16_t>(half + 1);  // round to nearest, ties up
    }
    return half;
}

float halfToFloat(uint16_t half) {
    uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1Fu;
    uint32_t mantissa = half & 0x3FFu;

    uint32_t bits;
    if (exponent == 0) {
        bits = sign;  // zero or subnormal half -> zero
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);  // inf/nan
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }

    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// --- Layout packing --------------------------------------------------------
// `a`/`b` are the logical, row-major fp32 matrices (A: M x K, B: K x N).

// gemm_kfirst expects B transposed: addr(k,n) = n*K + k.
std::vector<uint16_t> packBKFirst(const std::vector<float>& b, uint32_t K, uint32_t N) {
    std::vector<uint16_t> out(static_cast<size_t>(K) * N);
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t n = 0; n < N; ++n) {
            out[static_cast<size_t>(n) * K + k] = floatToHalf(b[static_cast<size_t>(k) * N + n]);
        }
    }
    return out;
}

// gemm_kfirst_tiled expects A packed into TILE_K-wide strips:
// addr(m,k) = (k/TILE_K)*M*TILE_K + m*TILE_K + k%TILE_K
std::vector<uint16_t> packAStripTiled(const std::vector<float>& a, uint32_t M, uint32_t K, uint32_t TILE_K) {
    std::vector<uint16_t> out(static_cast<size_t>(M) * K);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t strip = k / TILE_K;
            uint32_t kk = k % TILE_K;
            size_t addr = static_cast<size_t>(strip) * M * TILE_K + static_cast<size_t>(m) * TILE_K + kk;
            out[addr] = floatToHalf(a[static_cast<size_t>(m) * K + k]);
        }
    }
    return out;
}

// gemm_kfirst_tiled expects B packed into TILE_K-wide strips:
// addr(k,n) = (k/TILE_K)*N*TILE_K + n*TILE_K + k%TILE_K
std::vector<uint16_t> packBStripTiled(const std::vector<float>& b, uint32_t K, uint32_t N, uint32_t TILE_K) {
    std::vector<uint16_t> out(static_cast<size_t>(K) * N);
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t n = 0; n < N; ++n) {
            uint32_t strip = k / TILE_K;
            uint32_t kk = k % TILE_K;
            size_t addr = static_cast<size_t>(strip) * N * TILE_K + static_cast<size_t>(n) * TILE_K + kk;
            out[addr] = floatToHalf(b[static_cast<size_t>(k) * N + n]);
        }
    }
    return out;
}

struct PushConstants {
    uint32_t M, N, K;
};

struct GemmRunResult {
    double elapsedMillis;
    double worstErrorRatio;  // max over elements of |c - ref| / (atol + rtol*|ref|); pass when <= 1
    bool passed;
};

// Builds the pipeline for one shader variant, dispatches it once, times it,
// and verifies the result against `reference` (M x N, row-major, fp32).
GemmRunResult runGemmVariant(VulkanDevice& vulkanDevice, const std::string& shaderPath,
                             const std::vector<uint16_t>& packedA, const std::vector<uint16_t>& packedB,
                             uint32_t M, uint32_t N, uint32_t K,
                             uint32_t subgroupSize, uint32_t tileM, uint32_t tileN, uint32_t tileK,
                             const std::vector<float>& reference) {
    VkDevice device = vulkanDevice.getDevice();
    VkPhysicalDevice physicalDevice = vulkanDevice.getPhysicalDevice();

    const VkMemoryPropertyFlags hostVisible =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VulkanBuffer bufferA(device, physicalDevice, packedA.size() * sizeof(uint16_t),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostVisible);
    VulkanBuffer bufferB(device, physicalDevice, packedB.size() * sizeof(uint16_t),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostVisible);
    VulkanBuffer bufferC(device, physicalDevice, static_cast<VkDeviceSize>(M) * N * sizeof(uint16_t),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostVisible);

    bufferA.upload(packedA.data());
    bufferB.upload(packedB.data());

    VulkanPipeline pipeline(device);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // A
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // B
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // C
    };
    pipeline.createDescriptorSetLayout(bindings);

    // Specialization constants: id 0 is local_size_x (one workgroup == one
    // subgroup), ids 1-3 are TILE_M/TILE_N/TILE_K.
    uint32_t specData[4] = { subgroupSize, tileM, tileN, tileK };
    std::vector<VkSpecializationMapEntry> specEntries = {
        { 0, 0 * sizeof(uint32_t), sizeof(uint32_t) },
        { 1, 1 * sizeof(uint32_t), sizeof(uint32_t) },
        { 2, 2 * sizeof(uint32_t), sizeof(uint32_t) },
        { 3, 3 * sizeof(uint32_t), sizeof(uint32_t) },
    };
    std::vector<VkPushConstantRange> pushConstantRanges = {
        { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants) },
    };
    pipeline.createPipeline(shaderPath, "main", specEntries, specData, sizeof(specData),
                            pushConstantRanges, subgroupSize);

    std::vector<VkDescriptorPoolSize> poolSizes = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 } };
    pipeline.createDescriptorPool(poolSizes);
    VkDescriptorSet descriptorSet = pipeline.allocateDescriptorSet(pipeline.getDescriptorPool());

    VkDescriptorBufferInfo infoA = { bufferA.getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo infoB = { bufferB.getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo infoC = { bufferC.getBuffer(), 0, VK_WHOLE_SIZE };
    std::vector<VkWriteDescriptorSet> writes = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infoA, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infoB, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infoC, nullptr },
    };
    pipeline.updateDescriptorSet(descriptorSet, writes);

    VulkanTimer timer(device, physicalDevice, vulkanDevice.getComputeQueueFamilyIndex());
    VulkanCommandBuffer commandBuffer(device, vulkanDevice.getCommandPool());

    PushConstants pc{ M, N, K };

    commandBuffer.beginRecording();
    timer.begin(commandBuffer.getCommandBuffer());
    commandBuffer.pushConstants(pipeline.getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    // Per the shader headers: vkCmdDispatch(N / TILE_N, M / TILE_M, 1).
    commandBuffer.dispatchCompute(pipeline.getPipeline(), pipeline.getPipelineLayout(), descriptorSet,
                                  N / tileN, M / tileM, 1);
    timer.end(commandBuffer.getCommandBuffer());
    commandBuffer.endRecording();
    commandBuffer.submit(vulkanDevice.getComputeQueue());

    std::vector<uint16_t> packedC(static_cast<size_t>(M) * N);
    bufferC.download(packedC.data());

    // fp16 accumulation over K terms rounds at every step, so allow a
    // generous combined tolerance rather than a tight relative bound alone
    // (which blows up near zero-crossings of the reference value).
    const double atol = 0.1;
    const double rtol = 0.1;
    double worstRatio = 0.0;
    for (size_t i = 0; i < packedC.size(); ++i) {
        double c = halfToFloat(packedC[i]);
        double ref = reference[i];
        double allowed = atol + rtol * std::fabs(ref);
        double ratio = std::fabs(c - ref) / allowed;
        worstRatio = std::max(worstRatio, ratio);
    }

    GemmRunResult result;
    result.elapsedMillis = timer.getElapsedMillis();
    result.worstErrorRatio = worstRatio;
    result.passed = worstRatio <= 1.0;
    return result;
}

}  // namespace

int main(int argc, char** argv) try {
    (void)argc;
    (void)argv;

    // Step 1: Device with cooperative matrix support.
    VulkanDevice vulkanDevice(kDefaultEnableValidationLayers, true /*cooperative matrix*/);
    uint32_t subgroupSize = vulkanDevice.getSubgroupSize();

    // Step 2: Pick a float16 subgroup-scope cooperative matrix configuration.
    auto coopmatProperties = vulkanDevice.getCooperativeMatrixProperties();
    const VkCooperativeMatrixPropertiesKHR* chosen = nullptr;
    for (const auto& prop : coopmatProperties) {
        if (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR && prop.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            prop.scope == VK_SCOPE_SUBGROUP_KHR) {
            chosen = &prop;
            break;
        }
    }
    if (!chosen) {
        throw std::runtime_error("no float16 subgroup-scope VkCooperativeMatrixPropertiesKHR entry on this device");
    }
    uint32_t tileM = chosen->MSize, tileN = chosen->NSize, tileK = chosen->KSize;

    std::cout << "subgroupSize=" << subgroupSize << " TILE_M=" << tileM << " TILE_N=" << tileN
              << " TILE_K=" << tileK << std::endl;

    // Step 3: Problem size, a small multiple of each tile dimension so both
    // shaders' divisibility requirements hold regardless of the hardware's
    // actual tile sizes.
    const uint32_t multiplier = 8;
    uint32_t M = tileM * multiplier, N = tileN * multiplier, K = tileK * multiplier;

    // Step 4: Deterministic input data and CPU reference (computed in fp32
    // from the un-rounded inputs; the fp16 tolerance in runGemmVariant
    // accounts for the GPU's fp16 accumulation).
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> a(static_cast<size_t>(M) * K);
    std::vector<float> b(static_cast<size_t>(K) * N);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    std::vector<float> reference(static_cast<size_t>(M) * N, 0.0f);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < K; ++k) {
            float aVal = a[static_cast<size_t>(m) * K + k];
            for (uint32_t n = 0; n < N; ++n) {
                reference[static_cast<size_t>(m) * N + n] += aVal * b[static_cast<size_t>(k) * N + n];
            }
        }
    }

    // Step 5: Pack per-variant layouts and run each shader.
    std::vector<uint16_t> aPlain(static_cast<size_t>(M) * K);
    for (size_t i = 0; i < a.size(); ++i) aPlain[i] = floatToHalf(a[i]);
    std::vector<uint16_t> bKFirst = packBKFirst(b, K, N);

    std::vector<uint16_t> aTiled = packAStripTiled(a, M, K, tileK);
    std::vector<uint16_t> bTiled = packBStripTiled(b, K, N, tileK);

    GemmRunResult kfirst = runGemmVariant(vulkanDevice, vulkan::shaderPath("gemm_kfirst.spv"),
                                          aPlain, bKFirst, M, N, K, subgroupSize, tileM, tileN, tileK,
                                          reference);
    GemmRunResult tiled = runGemmVariant(vulkanDevice, vulkan::shaderPath("gemm_kfirst_tiled.spv"),
                                         aTiled, bTiled, M, N, K, subgroupSize, tileM, tileN, tileK,
                                         reference);

    // Step 6: Report.
    double gflops = 2.0 * M * N * K / 1e9;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "gemm_kfirst:       " << kfirst.elapsedMillis << " ms, "
              << (gflops / (kfirst.elapsedMillis / 1000.0)) << " GFLOP/s, "
              << (kfirst.passed ? "PASS" : "FAIL") << " (worst error ratio " << kfirst.worstErrorRatio << ")\n";
    std::cout << "gemm_kfirst_tiled: " << tiled.elapsedMillis << " ms, "
              << (gflops / (tiled.elapsedMillis / 1000.0)) << " GFLOP/s, "
              << (tiled.passed ? "PASS" : "FAIL") << " (worst error ratio " << tiled.worstErrorRatio << ")\n";
    std::cout << "tiled speedup: " << (kfirst.elapsedMillis / tiled.elapsedMillis) << "x\n";

    if (!kfirst.passed || !tiled.passed) {
        std::cerr << "GEMM verification failed." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;

} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
