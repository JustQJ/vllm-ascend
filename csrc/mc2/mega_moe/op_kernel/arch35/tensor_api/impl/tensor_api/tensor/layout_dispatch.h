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
    "impl/tensor_api/tensor/layout_dispatch.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file layout_dispatch.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_DISPATCH_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_DISPATCH_H

#include "impl/tensor_api/utils/utils_impl.h"

namespace AscendC {
namespace Te {

struct MakeTupleCons {
    template <typename... Ts>
    __aicore__ inline decltype(auto) operator()(Ts&&... ts) {
        return Std::make_tuple(Std::forward<Ts>(ts)...);
    }
};

template <typename F, typename T>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T&& t) {
    return t;
}

template <typename F, typename T0, typename T1>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T0&& t0, T1&& t1) {
    return f(t0, t1);
}

template <typename F, typename T0, typename T1, typename... Ts>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T0&& t0, T1&& t1, Ts&&... ts) {
    auto tuple1 = Make2Params2Tuple(f, t0, t1);
    auto tuple2 = Make2Params2Tuple(f, ts...);
    return Make2Params2Tuple(f, tuple1, tuple2);
}

template <typename... Ts>
__aicore__ inline decltype(auto) GetTuple(Ts&&... ts) {
    return Make2Params2Tuple(MakeTupleCons{}, ts...);
}

 template <typename T0, typename T1, typename T2, typename T3, typename... Ts> 
 __aicore__ inline decltype(auto) LayoutConstructor(T0&& t0, T1&& t1, T2&& t2, T3&& t3, Ts&&... ts) { 
     auto shape = GetTuple(t0, t1, t2, t3); 
     auto stride = GetTuple(ts...); 
     return Layout(shape, stride); 
 }

// layout_dispatch.h
template <LayoutFormat format, typename T>
struct LayoutDispatcher;

template <typename T>
struct LayoutDispatcher<LayoutFormat::NZ, T> {
    template <typename U, typename S>
    __aicore__ inline static decltype(auto) apply(U row, S column) {
        constexpr auto c0Ele = C0_ELEMENT<T>;
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(row, FRACTAL_FIXED), Std::Int<c0Ele>{},
                                 Std::ceil_division(column, c0Ele), Std::Int<c0Ele>{}, Std::Int<c0Ele * FRACTAL_FIXED>{},
                                 Std::Int<1>{}, c0Ele * Std::ceil_align(row, FRACTAL_FIXED));
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::NZ, Std::ignore_t> {
    template <typename U, typename S>
    __aicore__ inline static decltype(auto) apply(U row, S column) {
        constexpr auto c0Ele = C0_ELEMENT<uint16_t>;
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{},  Std::ceil_division(row, FRACTAL_FIXED), 
                                Std::Int<c0Ele>{},  Std::ceil_division(column, c0Ele), 
                                Std::Int<c0Ele>{},  Std::Int<c0Ele * FRACTAL_FIXED>{},
                                Std::Int<1>{},  c0Ele * Std::ceil_align(row, FRACTAL_FIXED)); 
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ZN, T> {
    template <typename U, typename S>
    __aicore__ inline static decltype(auto) apply(U row, S column) {
        constexpr auto c0Ele = C0_ELEMENT<T>;
        return LayoutConstructor(Std::Int<c0Ele>{},  Std::ceil_division(row, c0Ele),
                                Std::Int<FRACTAL_FIXED>{},  Std::ceil_division(column, FRACTAL_FIXED),
                                Std::Int<1>{},  c0Ele * Std::ceil_align(column, FRACTAL_FIXED),
                                Std::Int<c0Ele>{},  Std::Int<c0Ele * FRACTAL_FIXED>{});
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::DN, T> {
    template <typename U, typename S>
    __aicore__ inline static decltype(auto) apply(U row, S column) {
        return LayoutConstructor(Std::Int<1>{}, row, Std::Int<1>{}, column,
                                    Std::Int<0>{}, Std::Int<1>{}, Std::Int<0>{}, row);
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::DN, fp8_e8m0_t> {
    template <typename U, typename S>
    __aicore__ inline static decltype(auto) apply(U row, S column) {
        return LayoutConstructor(Std::Int<1>{}, row, Std::Int<2>{}, column / MX_SCALE_K0,
                                    Std::Int<0>{}, Std::Int<MX_SCALE_K0>{}, Std::Int<1>{}, MX_SCALE_K0 * row);
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ND, T> {
    template <typename U, typename S>
    __aicore__ inline static decltype(auto) apply(U row, S column) {
        return LayoutConstructor(Std::Int<1>{}, row, Std::Int<1>{}, column,
                                    Std::Int<0>{}, column, Std::Int<0>{}, Std::Int<1>{});
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::ND, fp8_e8m0_t> {
    template <typename U, typename S>
    __aicore__ inline static decltype(auto) apply(U row, S column) {
        return LayoutConstructor(Std::Int<2>{}, row / MX_SCALE_K0, Std::Int<1>{}, column,
                                    Std::Int<1>{}, MX_SCALE_K0 * column, Std::Int<0>{}, Std::Int<MX_SCALE_K0>{});
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ZZ, T> {
    template <typename U, typename S>
    __aicore__ inline static decltype(auto) apply(U row, S column) {
        constexpr auto c0Ele = C0_ELEMENT<T>;
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(row, FRACTAL_FIXED),
                                    Std::Int<c0Ele>{}, Std::ceil_division(column, c0Ele),
                                    Std::Int<c0Ele>{}, FRACTAL_FIXED * Std::ceil_align(column, c0Ele),
                                    Std::Int<1>{}, Std::Int<c0Ele * FRACTAL_FIXED>{});
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::ZZ, fp8_e8m0_t> {
    template <typename U, typename S>
    __aicore__ inline static decltype(auto) apply(U row, S column) {
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(row, FRACTAL_FIXED),
                                    Std::Int<MX_SCALE_K0>{}, column / MX_SCALE_K0,
                                    Std::Int<MX_SCALE_K0>{}, column * FRACTAL_FIXED,
                                    Std::Int<1>{}, Std::Int<C0_SIZE<>>{});
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::NN, fp8_e8m0_t> {
    template <typename U, typename S>
    __aicore__ inline static decltype(auto) apply(U row, S column) { // (scaleK, n)
        return LayoutConstructor(Std::Int<MX_SCALE_K0>{}, row / MX_SCALE_K0,
                                    Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(column, FRACTAL_FIXED),
                                    Std::Int<1>{}, Std::Int<C0_SIZE<>>{},
                                    Std::Int<MX_SCALE_K0>{}, row * FRACTAL_FIXED);
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_DISPTACH_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
