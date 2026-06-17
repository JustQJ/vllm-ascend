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
    "impl/tensor_api/arch/utils/is_format.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file is_format.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H
#define IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/tensor/pointer_pattern.h"
#include "impl/tensor_api/tensor/local_tensor_impl.h"

namespace AscendC {
namespace Te {

template <typename T, bool IsTuple>
struct ToTupleImpl;

template <typename T>
struct ToTupleImpl<T, true> {
    using type = T;
};

template <typename T>
struct ToTupleImpl<T, false> {
    using type = Std::tuple<T>;
};

template <typename T>
using ToTuple = ToTupleImpl<T, Std::is_tuple_v<T>>;

template <typename T>
struct GetTypeFromNDimTrait;

template <typename hPos, typename Pointer, 
          typename ShapeRows, typename ShapeCols, 
          typename StrideRows, typename StrideCols, 
          typename LayoutPattern>
struct GetTypeFromNDimTrait<
    LocalTensor<TensorAttribute<ViewEngine<HardwareMemPtr<hPos, Pointer>>,
        Layout<Shape<ShapeRows, ShapeCols>, Stride<StrideRows, StrideCols>, LayoutPattern>>>> {
    using ShapeRowTuple = typename ToTuple<ShapeRows>::type;
    using ShapeColTuple = typename ToTuple<ShapeCols>::type;
    using StrideRowTuple = typename ToTuple<StrideRows>::type;
    using StrideColTuple = typename ToTuple<StrideCols>::type;
    
    template <size_t Dim>
    using ShapeRowDim = typename Std::tuple_element<Dim, ShapeRowTuple>::type;
    
    template <size_t Dim>
    using ShapeColDim = typename Std::tuple_element<Dim, ShapeColTuple>::type;
    
    template <size_t Dim>
    using StrideRowDim = typename Std::tuple_element<Dim, StrideRowTuple>::type;
    
    template <size_t Dim>
    using StrideColDim = typename Std::tuple_element<Dim, StrideColTuple>::type;
};

namespace AttrInfo {
    struct Shape {};
    struct Stride {};
    struct Row {};
    struct Column {};
};

template <typename T, typename info1, typename info2, size_t dim>
struct GetNDimType;

template <typename T, size_t dim>
struct GetNDimType<T, AttrInfo::Shape, AttrInfo::Row, dim> {
    using type = Std::remove_cvref_t<typename GetTypeFromNDimTrait<T>::template ShapeRowDim<dim>>;
};

template <typename T, size_t dim>
struct GetNDimType<T, AttrInfo::Shape, AttrInfo::Column, dim> {
    using type = Std::remove_cvref_t<typename GetTypeFromNDimTrait<T>::template ShapeColDim<dim>>;
};

template <typename T, size_t dim>
struct GetNDimType<T, AttrInfo::Stride, AttrInfo::Row, dim> {
    using type = Std::remove_cvref_t<typename GetTypeFromNDimTrait<T>::template StrideRowDim<dim>>;
};

template <typename T, size_t dim>
struct GetNDimType<T, AttrInfo::Stride, AttrInfo::Column, dim> {
    using type = Std::remove_cvref_t<typename GetTypeFromNDimTrait<T>::template StrideColDim<dim>>;
};

template <typename TensorType, typename TargetLayoutPtn> 
struct IsSatisfiedPtnFormat {
    using LayoutPattern = GetLayoutPattern<typename TensorType::layoutType>;
    static constexpr bool value = Std::is_same_v<LayoutPattern, TargetLayoutPtn>;
};

template <typename TensorType, typename TargetLayoutPtn>
inline constexpr bool IsSatisfiedPtnFormatV = IsSatisfiedPtnFormat<TensorType, TargetLayoutPtn>::value;
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
