# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_MEGA_MOE_INC_OPTS)
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
    list(APPEND _MEGA_MOE_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
    list(APPEND _MEGA_MOE_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
    list(APPEND _MEGA_MOE_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
endif()

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        mega_moe_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MegaMoe
    OPTIONS
        --cce-auto-sync=off
        -DHCCL_COMM
        -Wno-c++20-extensions
        -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
        -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/tensor_api
        -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/tensor_api/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/tensor_api/impl
        ${_MEGA_MOE_INC_OPTS}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE mega_moe ACLNNTYPE aclnn_inner)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    )
endif()
