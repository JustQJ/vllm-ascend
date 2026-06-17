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
    "impl/tensor_api/utils/constant_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file constant_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H
#define IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H

#include "impl/tensor_api/utils/extra_impl.h"
#include "impl/tensor_api/utils/map_impl.h"

namespace AscendC {
namespace Te {
constexpr size_t TWO_DIM_DATA = 2;
constexpr size_t FOUR_DIM_DATA = 4;
constexpr size_t FRACTAL_FIXED = 16;
constexpr size_t MX_SCALE_K0 = 2;
constexpr uint32_t BLOCK_CUBE = 16;

struct ArchVersion {
    static constexpr uint32_t V3510 = 3510;
    static constexpr uint32_t V2201 = 2201;
};

struct GetArchVersion {
    __aicore__ inline constexpr uint32_t operator()() const {
#ifdef __NPU_ARCH__
        return __NPU_ARCH__;
#else
        return 0;
#endif
    }
};

constexpr uint32_t CURRENT_ARCH_VERSION = GetArchVersion{}();

namespace CopyMode {
    struct NORMAL {};
    struct NORMAL_COORD {};
    struct TRANS {};
    struct TRANS_COORD {};
    struct TRANS_B8B4 {};
    struct TRANS_B8B4_COORD {};
};

enum class LayoutFormat : uint8_t { NZ, ZN, ZZ, DN, ND, NN};

namespace Location {
    struct INVALID {};
    struct GM {};
    struct UB {};
    struct L1 {};
    struct L0A {};
    struct L0B {};
    struct L0C {};
    struct BIAS {};
    struct FIXBUF {};
}

template <typename TupleType>
using tuple_sequence = Std::make_index_sequence<Std::tuple_size_v<Std::remove_cvref_t<TupleType>>>;

template <typename T>
struct locationAttr {
    using gmAttr =          __gm__      T*;
    using cbufAttr =        __cbuf__    T*;
    using caAttr =          __ca__      T*;
    using cbAttr =          __cb__      T*;
    using ccAttr =          __cc__      T*;
    using ubufAttr =        __ubuf__    T*;
    using fbufAttr =        __fbuf__    T*;
    using ssbufAttr =       __ssbuf__   T*;
    using biasbufAttr =     __biasbuf__ T*;

    using type = Std::tuple<gmAttr, cbufAttr, caAttr, cbAttr, ccAttr, ubufAttr, fbufAttr, ssbufAttr, biasbufAttr>;

    using locationMap = TupleMap<Std::tuple<Location::GM, gmAttr>, Std::tuple<Location::L1, cbufAttr>, Std::tuple<Location::L0A, caAttr>,
    Std::tuple<Location::L0B, cbAttr>, Std::tuple<Location::L0C, ccAttr>, Std::tuple<Location::UB, ubufAttr>, 
    Std::tuple<Location::BIAS, biasbufAttr>, Std::tuple<Location::FIXBUF, fbufAttr>>;
};

template <typename A, typename ProcessedTuple>
struct AllElementsSameAsA;

template <typename A, typename First, typename... Rest>
struct AllElementsSameAsA<A, Std::tuple<First, Rest...>> {
    static constexpr bool value = Std::is_same_v<A, typename IterEle<First>::type> || AllElementsSameAsA<A, Std::tuple<Rest...>>::value;
};

template <typename A>
struct AllElementsSameAsA<A, Std::tuple<>> { static constexpr bool value = false; };

template <typename A, typename... BList>
struct CheckAllSame;

template <typename A, typename B, typename... RestB>
struct CheckAllSame<A, B, RestB...> {
    static constexpr bool value =  Std::is_same_v<A, B> || AllElementsSameAsA<A, typename locationAttr<B>::type>::value || CheckAllSame<A, RestB...>::value;
};

template <typename A>
struct CheckAllSame<A> { static constexpr bool value = false; };

template <typename A, typename... BList>
constexpr bool IsOneOfAttrV = CheckAllSame<A, BList...>::value;

template <typename DataType>
inline constexpr bool IsB4Type = IsOneOfAttrV<DataType, fp4x2_e1m2_t, fp4x2_e2m1_t>;

template<typename T = Std::ignore_t>
__aicore__ inline constexpr size_t GetC0Size() {
    constexpr size_t c0Size = 32;
    if constexpr (IsB4Type<T>) {
        return c0Size * 2;
    } else {
        return c0Size;
    }
}

template<typename T = Std::ignore_t>
constexpr size_t C0_SIZE = GetC0Size<T>();

template<typename T>
constexpr size_t C0_ELEMENT = C0_SIZE<T> / sizeof(T);

template <size_t N, typename = Std::make_index_sequence<N>>
struct EmptyGenerator;

template <size_t N, size_t... Idx>
struct EmptyGenerator<N, Std::index_sequence<Idx...>>
{   
    using type = Std::tuple<Std::Int<Idx * 0>...>;
};

template <size_t N>
struct TupleEmptyGenerator
{   
    static_assert((N > 0 && (N & 1) == 0), "N must be greater than 0, and must be even.");
    using type = Std::tuple<typename EmptyGenerator<N / 2>::type, 
        typename EmptyGenerator<N / 2>::type>;
};

template <size_t N>
using EmptyShape = typename TupleEmptyGenerator<N>::type;

template <size_t N>
using EmptyStride = typename TupleEmptyGenerator<N>::type;

template <size_t N>
using EmptyCoord = typename TupleEmptyGenerator<N>::type;

// IsIntegralConstant
template <typename T>
struct IsIntegralConstant : Std::false_type {};

template <size_t Value>
struct IsIntegralConstant<Std::Int<Value>> : Std::true_type {};

template <typename T>
constexpr bool IsIntegralConstantV = IsIntegralConstant<T>::value;

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
