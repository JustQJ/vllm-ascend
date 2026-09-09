# SPDX-License-Identifier: Apache-2.0
add_op_to_compiled_list()
set(TURBO_QUANT_GROUP_PREPARE_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for this operator")
if (BUILD_OPEN_PROJECT)
    set(turbo_quant_group_prepare_depends attention/turbo_quant_sparse_flash_attention
        CACHE INTERNAL "Shared query group planner header")
    target_sources(op_host_aclnn PRIVATE turbo_quant_group_prepare_def.cpp)
endif()
add_ops_compile_options(OP_NAME TurboQuantGroupPrepare OPTIONS --cce-auto-sync=off)
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE turbo_quant_group_prepare ACLNNTYPE aclnn)
endif()
