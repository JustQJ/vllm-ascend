/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_policy.h
 * \brief
 */
#pragma once

namespace Blaze {
namespace Gemm {
/* block schedule policies */
struct KernelMmadWithScale {}; // Multi-block with scale

/**
 * @struct MatmulWithScaleMx
 * @brief Mx Matrix multiplication with scaleA and scaleB
 */
template <uint64_t FULL_LOAD_MODE_ = 0>
struct MatmulWithScaleMx {
    using ScheduleType = KernelMmadWithScale;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
};

} // namespace Gemm
} // namespace Blaze
