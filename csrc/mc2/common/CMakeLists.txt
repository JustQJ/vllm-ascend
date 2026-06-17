# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    file(GLOB OPTILING_SRCS
        op_host/op_tiling/mc2_tiling_utils.cpp
        op_host/op_tiling/matmul_formulaic_tiling.cpp
        utils/mc2_hcom_topo_info.cpp
        utils/mc2_log.cpp
    )

    if(OPTILING_SRCS)
        add_tiling_modules()
        target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS})
    endif()
endif()
