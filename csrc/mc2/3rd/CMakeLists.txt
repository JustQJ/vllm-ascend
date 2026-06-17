# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB MC2_3RD_SRCS
    # mm
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/mat_mul_v3_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/op_tiling/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/op_tiling/arch35/*.cpp
    # qbmm
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_batch_matmul_v3/op_host/op_tiling/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_batch_matmul_v3/op_host/op_tiling/arch35/*.cpp
)

# MC2_COMPILE used to avoid tefusion_llt
# when blue use SUB_MC2_COMPILE when rty use MC2_COMPILE
if(MC2_3RD_SRCS AND (MC2_COMPILE OR SUB_MC2_COMPILE OR UT_TEST OR OP_HOST_UT))
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${MC2_3RD_SRCS})
endif()
