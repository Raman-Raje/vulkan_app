#!/usr/bin/env bash
# Recompiles the hand-written GLSL compute shaders in shaders/ to shaders/spv/.
#
# tvmgen_default_fused_layout_transform_kernel0_spv.comp is not compiled here:
# it is the GLSL *disassembly* of a TVM-generated SPIR-V module, not a source
# file, and it does not compile as valid GLSL.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shader_dir="$repo_root/shaders"
out_dir="$shader_dir/spv"

if ! command -v glslc >/dev/null 2>&1; then
    echo "error: glslc not found (install the Vulkan SDK)" >&2
    exit 1
fi

mkdir -p "$out_dir"

for shader in buffer_add buffer_image buffers images_add; do
    src="$shader_dir/$shader.comp"
    out="$out_dir/$shader.spv"
    echo "glslc $shader.comp -> spv/$shader.spv"
    glslc -fshader-stage=compute "$src" -o "$out"
done

echo "done"
