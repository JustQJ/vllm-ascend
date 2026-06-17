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
    "impl/tensor_api/arch/vector/cast/cast_vf.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file cast_vf.h
* \brief
*/
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_CAST_CAST_VF_H
#define IMPL_TENSOR_API_ARCH_VECTOR_CAST_CAST_VF_H

#include "impl/tensor_api/arch/vector/cast/instruction.h"
#include "impl/tensor_api/arch/vector/utils/mask_utils.h"

namespace AscendC {
namespace Te {

template<typename CalcFunc, typename TraitType>
class CastVF {
public:
    template<typename T, typename U>
    __simd_vf__ inline static void Run(__ubuf__ T* dst, __ubuf__ U* src, uint16_t repeat, uint16_t oneRepSize, uint32_t dataSize)
    {
        using DstRegType = typename VectorTypeTransform::template Get<T>;
        using SrcRegType = typename VectorTypeTransform::template Get<U>;
        using greaterType = Std::conditional_t<(sizeof(T) > sizeof(U)), T, U>;

        DstRegType dstReg;
        SrcRegType srcReg;

        vector_bool dstMask;
        vector_bool fullMask = Inst::CreateMask::template Run<uint8_t, decltype(PAT_ALL)>();

        for (uint16_t i = 0; i < repeat; i++) {
            dstMask = Inst::UpdateMask::template Run<greaterType>(dataSize);
            asc_loadalign(srcReg, src + i * oneRepSize);
            CalcFunc::template Run(dstReg, srcReg, fullMask);
            asc_storealign(dst + i * oneRepSize, dstReg, dstMask);
        }
    }
};

template<typename CalcFunc, typename TraitType>
class Transform2CastVF {
public:
    template<typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src)
    {
        using dstType = GetAttributeElementType<typename T::elementType*>;
        using srcType = GetAttributeElementType<typename U::elementType*>;

        uint32_t dstSize = dst.Size();

        uint16_t VECTOR_REG_WIDTH = asc_get_vf_len();
        using greaterType = Std::conditional_t<(sizeof(dstType) > sizeof(srcType)), dstType, srcType>;
        uint16_t oneRepSize = VECTOR_REG_WIDTH / sizeof(greaterType);

        uint16_t repeat = Std::ceil_division(dstSize, oneRepSize);
        
        CastVF<CalcFunc, TraitType>::template Run(dst.Data().Get(), src.Data().Get(), repeat, oneRepSize, dstSize);
    }
};

namespace CastRoundMode {
struct Rint {};
struct Round {};
struct Floor {};
struct Ceil {};
struct Trunc {};
}

namespace CastSatMode {
struct Sat {};
struct NoSat {};
}

namespace CastIndexPos {
struct Odd {};
struct Even {};
struct PartP1 {};
struct PartP2 {};
struct PartP3 {};
}

template<typename T, typename U>
struct CastTrait2VF {
    using srcType = GetAttributeElementType<typename U::elementType*>;
    using dstType = GetAttributeElementType<typename T::elementType*>;
    using roundMode = Std::ignore_t;
    using satMode = Std::ignore_t;
    using indexPos = Std::ignore_t;
};

using CastTrait2CastFunc = TupleMap<
    Std::tuple<Std::tuple<half, half, Inst::Ceil, Std::ignore_t, Std::ignore_t>, Inst::Ceil>,
    Std::tuple<Std::tuple<bfloat16_t, bfloat16_t, Inst::Ceil, Std::ignore_t, Std::ignore_t>, Inst::Ceil>,
    Std::tuple<Std::tuple<float, float, Inst::Ceil, Std::ignore_t, Std::ignore_t>, Inst::Ceil>,
    Std::tuple<Std::tuple<uint8_t, uint16_t, Std::ignore_t, Std::ignore_t, Std::ignore_t>, Inst::U82U16>>;


template<typename CalcFunc, typename TraitType>
class Transform2CastInstMatch {
public:
    template<typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src)
    {
        using trait = Std::conditional_t<Std::is_same_v<TraitType, Std::ignore_t>, CastTrait2VF<T, U>, TraitType>;
        using roundMode = Std::conditional_t<Std::is_same_v<typename trait::srcType, typename trait::dstType>, CalcFunc, typename trait::roundMode>;

        using condition = Std::tuple<typename trait::srcType, typename trait::dstType, 
            roundMode, typename trait::satMode, typename trait::indexPos>;

        using func = CastTrait2CastFunc::template Get<condition>;
        Transform2CastVF<func, TraitType>::template Run(dst, src);
    }
};

}
}

#endif // IMPL_TENSOR_API_ARCH_VECTOR_CAST_CAST_VF_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
