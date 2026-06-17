/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "impl/tensor_api/arch/datamove/l0c_to_gm/npu_arch_3510/data_copy.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file data_copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_GM_NPU_ARCH_3510_DATA_COPY_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_GM_NPU_ARCH_3510_DATA_COPY_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/arch/datamove/l1_to_fb/copy.h"
#include "impl/tensor_api/arch/datamove/common/instruction.h"
#include "impl/tensor_api/arch/datamove/l0c_to_gm/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

struct CopyL0C2GMTrait {
    __aicore__ constexpr CopyL0C2GMTrait() {}

    __aicore__ constexpr CopyL0C2GMTrait(RoundMode roundModeIn, bool enableReluIn, bool enableChannelSplitIn) :
        roundMode(roundModeIn), enableRelu(enableReluIn), enableChannelSplit(enableChannelSplitIn)
    {}

    RoundMode roundMode = RoundMode::DEFAULT;
    bool enableRelu = false;
    bool enableChannelSplit = false;
};

class DataCopyL0C2GM3510 {
public:
    template <const CopyL0C2GMTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src, const FixpipeParams& params)
    {
        constexpr QuantMode_t quantPre = GetQuantMode<trait.roundMode, T, U>();
        CheckTemplate<trait, quantPre, T, U>();
        CheckDataTypeFor3510::CheckL0C2GmDataType<quantPre, T, U>();
        SetRegisterImpl<T, U>(dst, src);
        DataCopyImpl<trait, quantPre, T, U>(dst, src, params);
    }

    template <const CopyL0C2GMTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src, uint64_t quant, const FixpipeParams& params)
    {
        constexpr QuantMode_t quantPre = GetQuantMode<trait.roundMode, T, U, uint64_t>();
        CheckTemplate<trait, quantPre, T, U>();
        SetRegisterImpl<T, U>(dst, src, quant);
        DataCopyImpl<trait, quantPre, T, U>(dst, src, params);
    }

private:
    template <const CopyL0C2GMTrait& trait, QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckLayoutPattern<U, T>();
    }

    template <const CopyL0C2GMTrait& trait, QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src, const FixpipeParams& params)
    {
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        uint32_t nSize = Std::min(GetTotalColumnShape(srcLayout), GetTotalColumnShape(dstLayout));
        uint32_t mSize = Std::min(GetTotalRowShape(srcLayout), GetTotalRowShape(dstLayout));
        uint32_t srcStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = 0;

        if constexpr (IsSatisfiedPtnFormatV<T, NDExtLayoutPtn>) {
            dstStride = GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(dstLayout);
        } else {
            dstStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(dstLayout);
        }

        bool reluEn = trait.enableRelu;
        uint8_t unitFlag = params.unitFlag;
        bool nz2ndEn = IsSatisfiedPtnFormatV<T, NDExtLayoutPtn>;
        bool nz2dnEn = IsSatisfiedPtnFormatV<T, DNExtLayoutPtn>;

        uint8_t cacheMode = dst.Engine().GetCacheMode();
        bool isChannelSplit = trait.enableChannelSplit;

        CopyMatrixCcToGmInstr::DataCopy<quantPre, T, U>(dst, src, nSize, mSize, srcStride, dstStride, cacheMode, reluEn,
                                                        unitFlag, isChannelSplit, nz2ndEn, nz2dnEn);
    }
};

class DataCopyL0C2GMVectorQuant3510 {
public:
    template <const CopyL0C2GMTrait& trait, typename T, typename U, typename V>
    __aicore__ inline static void Run(const T& dst, const U& src, const V& quant, const FixpipeParams& params)
    {
        constexpr QuantMode_t quantPre = GetQuantMode<trait.roundMode, T, U, V>();
        CheckTemplate<trait, T, U>();
        SetRegisterImpl<T, U>(dst, src);
        DataCopyImpl<trait, quantPre, T, U, V>(dst, src, quant, params);
    }

private:
    template <const CopyL0C2GMTrait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckLayoutPattern<U, T>();
    }

    template <const CopyL0C2GMTrait& trait, typename T, typename U, bool IsTail>
    __aicore__ inline static auto GenerateParams(const T& dst, const U& src, const FixpipeParams& params)
    {
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        uint32_t nSize = GetTotalColumnShape(srcLayout);
        if constexpr (IsTail) {
            nSize %= MAIN_LOOP_N_SIZE_3510;
        } else {
            if (nSize > MAIN_LOOP_N_SIZE_3510) {
                nSize = MAIN_LOOP_N_SIZE_3510;
            }
        }

        const uint32_t mSize = GetTotalRowShape(srcLayout);
        const uint32_t srcStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = 0;
        if constexpr (IsSatisfiedPtnFormatV<T, NDExtLayoutPtn>) {
            dstStride = GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(dstLayout);
        } else {
            dstStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(dstLayout);
        }

        const bool reluEnable = trait.enableRelu;
        const uint8_t unitFlag = params.unitFlag;

        constexpr bool nz2ndEnable = IsSatisfiedPtnFormatV<T, NDExtLayoutPtn>;
        constexpr bool nz2dnEnable = IsSatisfiedPtnFormatV<T, DNExtLayoutPtn>;

        const uint8_t cacheMode = dst.Engine().GetCacheMode();
        const bool channelSplit = trait.enableChannelSplit;
        return Std::make_tuple(nSize, mSize, srcStride, dstStride, cacheMode, reluEnable, unitFlag, channelSplit,
                               nz2ndEnable, nz2dnEnable);
    }

    template <typename T>
    __aicore__ inline static void CopyL12Fb(const T& src, uint16_t calNSize, uint16_t nIterIndex)
    {
        auto dstAddr = reinterpret_cast<__fbuf__ uint64_t*>(AllocFbTempBuf(calNSize));
        auto dst = MakeTensor(MakeMemPtr<Location::FIXBUF>(dstAddr), src.Layout());
        auto coord = MakeCoord(Std::Int<0>{}, nIterIndex * MAIN_LOOP_N_SIZE_3510);
        auto shape = MakeShape(Std::Int<1>{}, calNSize);
        auto tileSrc = src.Slice(coord, shape);
        DataCopyL12FB3510::Run<DEFAULT_COPY_L1_FB_TRAIT>(dst, tileSrc);
        SetFpc(dstAddr);
    }

    template <QuantMode_t quantPre, typename T, typename U, typename ParamTuple, size_t... Is>
    __aicore__ inline static void DataCopyWrapper(const T& dst, const U& src, const ParamTuple& paramTuple,
                                                  Std::index_sequence<Is...>)
    { CopyMatrixCcToGmInstr::DataCopy<quantPre>(dst, src, Std::get<Is>(paramTuple)...); }

    template <const CopyL0C2GMTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V>
    __aicore__ inline static void ExecuteDataCopy(const T& dst, const U& src, const V& quant, uint16_t nIterNum,
                                                  uint32_t calNSize, uint32_t tailNSize, const FixpipeParams& params)
    {
        const auto mainLoopParam = GenerateParams<trait, T, U, false>(dst, src, params);

        for (uint16_t i = 0; i < nIterNum; ++i) {
            CopyL12Fb(quant, calNSize, i);
            InsertSync();

            const auto srcCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, i * CBURST_NUM_3510));
            const auto dstCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, i * MAIN_LOOP_N_SIZE_3510));

            DataCopyWrapper<quantPre>(dst(dstCoord), src(srcCoord), mainLoopParam,
                                      Std::make_index_sequence<Std::tuple_size_v<decltype(mainLoopParam)>>{});
        }

        if (tailNSize) {
            const auto tailParam = GenerateParams<trait, T, U, true>(dst, src, params);

            CopyL12Fb(quant, tailNSize, nIterNum);
            InsertSync();

            const auto srcCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, nIterNum * CBURST_NUM_3510));
            const auto dstCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, nIterNum * MAIN_LOOP_N_SIZE_3510));

            DataCopyWrapper<quantPre>(dst(dstCoord), src(srcCoord), tailParam,
                                      Std::make_index_sequence<Std::tuple_size_v<decltype(tailParam)>>{});
        }
    }

    template <const CopyL0C2GMTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src, const V& quant, const FixpipeParams& params)
    {
        uint32_t nSize = GetTotalColumnShape(src.Layout());
        uint16_t nIterNum = 1;
        uint32_t calNSize = nSize;
        uint32_t tailNSize = 0;
        if (calNSize > MAIN_LOOP_N_SIZE_3510) {
            nIterNum = nSize / MAIN_LOOP_N_SIZE_3510;
            tailNSize = nSize % MAIN_LOOP_N_SIZE_3510;
            calNSize = MAIN_LOOP_N_SIZE_3510;
        }
        ExecuteDataCopy<trait, quantPre>(dst, src, quant, nIterNum, calNSize, tailNSize, params);
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_GM_NPU_ARCH_3510_DATA_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif