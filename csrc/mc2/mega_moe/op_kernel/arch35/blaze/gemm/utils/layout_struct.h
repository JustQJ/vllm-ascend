/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {

// Layout for 8-bit weight tiles in Unified Buffer.
//
// This is a specialized UB layout for 8-bit weight data.  Unlike the standard
// FRACTAL_FIXED=16, this uses N0=VEC_REG_ELEM/C0=8 because the weight data is
// produced by Vector unit writes in 256-element (VEC_REG_ELEM) chunks.
// The InnerStride template parameter exposes inter-chunk spacing control to
// the MTE3 copy path, which is not expressible with built-in layout formulas.
template <typename T, uint64_t InnerStride>
struct Weight8BitUBLayout {
    static_assert(std::is_same_v<T, __fp8e4m3>, "Weight8BitUBLayout expects __fp8e4m3");

    __aicore__ inline decltype(auto) operator()(int64_t kSize, int64_t nSize)
    {
        static constexpr int64_t C0 = 32;
        static constexpr int64_t VEC_REG_ELEM = 256;
        static constexpr int64_t N0 = VEC_REG_ELEM / C0;
        static constexpr int64_t STRIDE_UNIT = 1;
        int64_t k1 = CeilDiv(kSize, C0);
        int64_t n1 = CeilDiv(nSize, N0);

        // Shape: ((C0, k1), (N0, n1))
        auto shape = AscendC::Te::MakeShape(
            AscendC::Te::MakeShape(AscendC::Std::Int<C0>{}, k1),
            AscendC::Te::MakeShape(AscendC::Std::Int<N0>{}, n1));
        // Stride: (MakeStride(1, n1 * InnerStride), MakeStride(C0, InnerStride))
        auto stride = AscendC::Te::MakeStride(
            AscendC::Te::MakeStride(AscendC::Std::Int<STRIDE_UNIT>{},
                                    n1 * AscendC::Std::Int<InnerStride>{}),
            AscendC::Te::MakeStride(AscendC::Std::Int<C0>{},
                                    AscendC::Std::Int<InnerStride>{}));
        return AscendC::Te::MakeLayout(shape, stride);
    }
};

} // namespace Gemm
} // namespace Blaze

