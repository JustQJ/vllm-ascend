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
 * \file copy_scale_l1_to_l0b.h
 * \brief
 */
#pragma once

#include "tensor_api/include/tensor_api/tensor.h"
#include "../utils/common_utils.h"
#include "c_api/asc_simd.h"

namespace Blaze::Gemm::Tile {
struct CopyL12L0MxScaleB3510 {
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        static_assert(
            AscendC::Std::is_one_of_v<
                AscendC::Std::tuple<dstType, srcType>, AscendC::Std::tuple<__cb__ fp8_e8m0_t, __cbuf__ fp8_e8m0_t>>,
            "The data type is not supported.");
        // (n1, k/64, n0, 2)
        // shape ((2, k/64), (n0, n1))
        // stride ((2, k/64*n0*2), (1, n0*2))
        // Nn -> Nn
        uint16_t nStartPosition = Blaze::Gemm::CeilDiv(AscendC::Std::get<1>(coord), MATMUL_MNK_ALIGN);
        uint16_t kStartPosition = Blaze::Gemm::CeilDiv(AscendC::Std::get<0>(coord), 2);
        auto nStep = AscendC::Std::get<1>(AscendC::Std::get<1>(dst.Layout().Shape()));
        auto kStep = AscendC::Std::get<1>(AscendC::Std::get<0>(dst.Layout().Shape()));
        auto srcStride = AscendC::Std::get<1>(AscendC::Std::get<1>(src.Layout().Stride())) >> 5;
        auto dstStride = kStep;
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.Data().Get())) >> 4;
        asc_copy_l12l0b_mx(
            mxDstAddr, src.Data().Get(), nStartPosition, kStartPosition, nStep, kStep, srcStride, dstStride);
    }
};

} // namespace Blaze::Gemm::Tile

template <>
struct AscendC::Te::CopyTraits<Blaze::Gemm::Tile::CopyL12L0MxScaleB3510>
    : public CopyTraits<
          Blaze::Gemm::Tile::CopyL12L0MxScaleB3510, CopyL12L0ATraitDefault, Blaze::Gemm::Tile::CopyL12L0MxScaleB3510,
          CopyL12L0ATraitDefault> {};
