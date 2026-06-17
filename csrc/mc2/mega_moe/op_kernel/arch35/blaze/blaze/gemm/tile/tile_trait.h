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
 * \file tile_trait.h
 * \brief
 */
#pragma once

#include "tensor_api/tensor.h"

namespace Blaze::Gemm::Tile {

constexpr AscendC::Te::MmadTrait MX_MMAD_TRAIT =
    AscendC::Te::MmadTrait{0, false, false, true, AscendC::Te::MmadType::MX};
struct MmadTraitMX {
    using TraitType = AscendC::Te::MmadTrait;
    static constexpr const TraitType value = MX_MMAD_TRAIT;
};

} // namespace Blaze::Gemm::Tile
