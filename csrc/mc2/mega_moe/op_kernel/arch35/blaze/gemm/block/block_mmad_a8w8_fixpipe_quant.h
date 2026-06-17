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
 * \file block_mmad_a8w8_fixpipe_quant.h
 * \brief A8W8 quantized matmul block with fixpipe dequant (Tensor API)
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::SetFlag;
using AscendC::WaitFlag;
using AscendC::SetMMLayoutTransform;
using AscendC::TOTAL_L0A_SIZE;
using AscendC::TOTAL_L0C_SIZE;
using AscendC::TOTAL_L1_SIZE;

template <
    uint64_t A_FULL_LOAD_MODE, bool ATOMIC_ADD, class AType_, class LayoutA_, class BTypeTuple_, class LayoutB_,
    class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<
    MatmulWithScaleFixpipeQuant<A_FULL_LOAD_MODE, ATOMIC_ADD>, AType_, LayoutA_, BTypeTuple_, LayoutB_, CType_, LayoutC_,
    BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = typename AscendC::Std::tuple_element<0, BTypeTuple_>::type;
    using CType = CType_;
    using BiasType = BiasType_;
    using X2ScaleType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using L0CType = typename AscendC::GetMmDstType<AType>::Type;
    using DispatchPolicy = MatmulWithScaleFixpipeQuant<A_FULL_LOAD_MODE, ATOMIC_ADD>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool weightNz = IsWeightNz<LayoutB>::value;
    static constexpr bool transA = IsTrans<LayoutA>::value;
    static constexpr bool transB = IsTrans<LayoutB>::value;
    static constexpr int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    static constexpr uint16_t L0C_C0 = 16;
    static constexpr uint16_t SCALE_BUFFER_NUM = 2;
    static constexpr uint16_t AB_L1_TWO_BUFFER = 2;
    static constexpr uint16_t BIAS_BUFFER_FLAG_0 = 4;
    static constexpr uint16_t BIAS_BUFFER_FLAG_1 = 5;
    static constexpr uint16_t X2_SCALE_BUFFER_FLAG_0 = 0;
    static constexpr uint16_t X2_SCALE_BUFFER_FLAG_1 = 1;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    __aicore__ inline BlockMmad()
    {
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        SetFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0);
        SetFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_1);
        SetFlag<HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_0);
        SetFlag<HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_1);
        SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        WaitFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0);
        WaitFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_1);
        WaitFlag<HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_0);
        WaitFlag<HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_1);
        SetMMLayoutTransform(false);
    }

    __aicore__ inline void Init(
        const ProblemShape& problemShape, const BlockShape& l0TileShape, const uint64_t& kAL1, const uint64_t& kBL1,
        const uint64_t& l1BufferNum, QuantMode quantMode, bool isBias, bool dbL0C)
    {
        m_ = AscendC::Te::Get<IDX_M_IDX>(problemShape);
        n_ = AscendC::Te::Get<IDX_N_IDX>(problemShape);
        k_ = AscendC::Te::Get<IDX_K_IDX>(problemShape);
        baseM_ = AscendC::Te::Get<IDX_M_IDX>(l0TileShape);
        baseN_ = AscendC::Te::Get<IDX_N_IDX>(l0TileShape);
        baseK_ = AscendC::Te::Get<IDX_K_IDX>(l0TileShape);
        isBias_ = isBias;
        l1BufNum_ = l1BufferNum;
        enableL0cPingPong_ = dbL0C;
        if (quantMode == QuantMode::PERCHANNEL_MODE) {
            x2ScaleL1OneBuffer_ = baseN_ * sizeof(uint64_t);
        }
        if (isBias_) {
            biasL1OneBuffer_ = baseN_ * sizeof(BiasType);
        }
        if constexpr (DispatchPolicy::fullLoadMode != 0) {
            kBL1_ = kBL1;
            kAL1_ = kBL1_;
            kL1_ = kBL1;
            uint64_t mAlign = CeilAlign(baseM_, transA ? static_cast<uint64_t>(C0_SIZE) : BLOCK_CUBE);
            uint64_t kAlign = CeilAlign(k_, transA ? BLOCK_CUBE : static_cast<uint64_t>(C0_SIZE));
            aL1OneBuffer_ = mAlign * kAlign;
            bL1OneBuffer_ = baseN_ * kL1_;
            kL1Iter_ = CeilDiv(k_, kL1_);
        } else {
            if (l1BufferNum == AB_L1_TWO_BUFFER) {
                kAL1_ = kAL1;
                kBL1_ = kBL1;
                aL1OneBuffer_ = baseM_ * kAL1_;
                bL1OneBuffer_ = baseN_ * kBL1_;
                kAL1Iter_ = CeilDiv(k_, kAL1_);
                kBL1Iter_ = CeilDiv(k_, kBL1_);
                if (kAL1 == kBL1) {
                    kL1_ = kAL1;
                    kL1Iter_ = CeilDiv(k_, kL1_);
                }
            } else {
                kL1_ = Min(kAL1, kBL1);
                aL1OneBuffer_ = baseM_ * kL1_;
                bL1OneBuffer_ = baseN_ * kL1_;
                kL1Iter_ = CeilDiv(k_, kL1_);
                kAL1_ = kL1_;
                kBL1_ = kL1_;
            }
        }
        GetL1BufferOffset();
    }

    template <typename TensorA, typename TensorB, typename TScale, typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(
        TensorA gmA, TensorB gmB, TScale scaleGlobal, TensorBias gmBias, TensorC gmC, BlockShape singleShape)
    {
        uint64_t curML1 = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        uint64_t curNL1 = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        AscendC::Te::MmadParams mmadParams;
        mmadParams.m = static_cast<uint16_t>(curML1);
        mmadParams.n = static_cast<uint16_t>(curNL1);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        uint16_t scaleL1BufId = scaleLoopCnt_ & 1;
        uint16_t biasBufId = biasLoopCnt_ & 1;

        auto layoutL0C = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(curML1, curNL1);
        auto c1Local = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, L0CType>(l0cOffset), layoutL0C);

        constexpr bool isScalarScale = IsSameType<TScale, uint64_t>::value;
        constexpr bool isPerChannelScale = AscendC::Te::IsAttrTensorV<TScale>;

        auto layoutX2L1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<uint64_t>>(1, curNL1);
        auto tensorX2L1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, uint64_t>(l1BufferX2ScaleOffset_[scaleL1BufId]), layoutX2L1);

        if constexpr (isScalarScale) {
            scalarScale_ = scaleGlobal;
        } else if constexpr (isPerChannelScale && !IsSameType<CType, int32_t>::value) {
            WaitFlag<HardEvent::FIX_MTE2>(scaleL1BufId);
            auto copyGM2L1Scale = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            AscendC::Te::Copy(copyGM2L1Scale, tensorX2L1, scaleGlobal);
            SetFlag<HardEvent::MTE2_FIX>(scaleL1BufId);
            WaitFlag<HardEvent::MTE2_FIX>(scaleL1BufId);
        }

        auto layoutBiasL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<BiasType>>(1UL, isBias_ ? curNL1 : 1UL);
        auto tensorBiasL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(isBias_ ? l1BufferBiasOffset_[biasBufId] : 0UL), layoutBiasL1);
        if (isBias_) {
            WaitFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0 + biasBufId);
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            AscendC::Te::Copy(copyGM2L1, tensorBiasL1, gmBias);
            SetFlag<HardEvent::MTE2_MTE1>(BIAS_BUFFER_FLAG_0 + biasBufId);
            WaitFlag<HardEvent::MTE2_MTE1>(BIAS_BUFFER_FLAG_0 + biasBufId);
        }

        if (kAL1_ == kBL1_) {
            IterateABL1(gmA, gmB, mmadParams, c1Local, curML1, curNL1, tensorBiasL1, biasBufId);
        } else if (kAL1_ > kBL1_) {
            IterateAL1BL1(gmA, gmB, mmadParams, c1Local, curML1, curNL1, tensorBiasL1, biasBufId);
        } else {
            IterateBL1AL1(gmA, gmB, mmadParams, c1Local, curML1, curNL1, tensorBiasL1, biasBufId);
        }

        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        if constexpr (IsSameType<CType, int32_t>::value) {
            AscendC::Te::Copy(copyL0C2GM.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), gmC, c1Local);
        } else if constexpr (isScalarScale) {
            AscendC::Te::Copy(copyL0C2GM.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), gmC, c1Local, scalarScale_);
        } else if constexpr (isPerChannelScale) {
            AscendC::Te::Copy(copyL0C2GM.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), gmC, c1Local, tensorX2L1);
        }

        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
        if constexpr (isPerChannelScale && !IsSameType<CType, int32_t>::value) {
            SetFlag<HardEvent::FIX_MTE2>(scaleL1BufId);
            scaleLoopCnt_++;
        }
        if (isBias_) {
            SetFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0 + biasBufId);
            biasLoopCnt_++;
        }
    }

private:
    static constexpr uint32_t L0_INIT_MODE_ABL1 = 0U;
    static constexpr uint32_t L0_INIT_MODE_SPLIT = 1U;

    __aicore__ inline void GetL1BufferOffset()
    {
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            for (uint16_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                uint64_t l1Offset = (TOTAL_L1_SIZE >> 1) * (bufferId & 1);
                l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
                l1BufferBOffset_[bufferId] =
                    l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            for (uint16_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                l1BufferX2ScaleOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                l1BufferBiasOffset_[bufferId] = l1BufferX2ScaleOffset_[bufferId] + x2ScaleL1OneBuffer_;
            }
        } else {
            uint64_t mAlign = CeilAlign(baseM_, transA ? static_cast<uint64_t>(C0_SIZE) : BLOCK_CUBE);
            uint64_t kAlign = CeilAlign(k_, transA ? BLOCK_CUBE : static_cast<uint64_t>(C0_SIZE));
            aL1OneBuffer_ = mAlign * kAlign;
            l1BufferAOffset_[0] = bL1OneBuffer_ * (l1BufNum_ >> 1) + x2ScaleL1OneBuffer_ + biasL1OneBuffer_;
            uint64_t b1Offset = l1BufferAOffset_[0] + aL1OneBuffer_ >= (TOTAL_L1_SIZE >> 1)
                                    ? l1BufferAOffset_[0] + aL1OneBuffer_
                                    : (TOTAL_L1_SIZE >> 1);
            for (uint16_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            for (uint16_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                l1BufferX2ScaleOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                l1BufferBiasOffset_[bufferId] = l1BufferX2ScaleOffset_[bufferId] + x2ScaleL1OneBuffer_;
            }
        }
    }

    template <typename TensorAL1, typename TensorBL1, typename TensorBiasL1, typename C1Tensor>
    __aicore__ inline void Iterate(
        AscendC::Te::MmadParams& mmadParams, TensorAL1 tensorAL1, TensorBL1 tensorBL1, const TensorBiasL1& tensorBiasL1, C1Tensor& c1Local,
        uint64_t curML1, uint64_t curNL1, uint64_t curInnerKL1, uint64_t aKPrefix, uint64_t bKPrefix, bool isL1LastRound,
        uint32_t initMode, uint64_t kL1OuterIdx, uint64_t kL1InnerIdx, uint16_t biasBufId)
    {
        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        uint64_t kL0Iter = CeilDiv(curInnerKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            uint64_t curKL0 = (iter1 == kL0Iter - 1) ? (curInnerKL1 - iter1 * baseK_) : baseK_;
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);

            auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(curML1, curKL0);
            auto layoutBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(curKL0, curNL1);
            auto l0aLocal = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
            auto l0bLocal = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);

            WaitFlag<HardEvent::M_MTE1>(l0PingPong_ & 0x1);

            auto aL1Sub = tensorAL1.Slice(AscendC::Te::MakeCoord(0, aKPrefix + iter1 * baseK_), AscendC::Te::MakeShape(curML1, curKL0));
            AscendC::Te::Copy(copyL12L0A, l0aLocal, aL1Sub);

            bool needBias = isBias_ && kL1OuterIdx == 0 && kL1InnerIdx == 0 && iter1 == 0;
            auto bL1Sub = tensorBL1.Slice(AscendC::Te::MakeCoord(bKPrefix + iter1 * baseK_, 0), AscendC::Te::MakeShape(curKL0, curNL1));
            AscendC::Te::Copy(copyL12L0B, l0bLocal, bL1Sub);

            // tensorBt construction is pure address/layout math; the data movement only happens when needBias.
            auto layoutBt = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, CeilAlign(curNL1, BLOCK_CUBE));
            auto tensorBt = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, BiasType>(baseN_ * biasBufId * sizeof(BiasType)), layoutBt);
            if (needBias) {
                auto copyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
                AscendC::Te::Copy(copyL12BT, tensorBt, tensorBiasL1);
            }

            SetFlag<HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            WaitFlag<HardEvent::MTE1_M>(l0PingPong_ & 0x1);

            mmadParams.k = static_cast<uint16_t>(curKL0);
            mmadParams.unitFlag = (isL1LastRound && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            mmadParams.cmatrixInitVal = (initMode == L0_INIT_MODE_ABL1)
                                            ? (kL1OuterIdx == 0 && iter1 == 0)
                                            : (kL1OuterIdx == 0 && kL1InnerIdx == 0 && iter1 == 0);

            using MmadAtomT =
                AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, AscendC::Te::MmadTraitDefault>>;
            if (needBias) {
                AscendC::Te::Mmad(MmadAtomT{}.with(mmadParams), c1Local, l0aLocal, l0bLocal, tensorBt);
            } else {
                AscendC::Te::Mmad(MmadAtomT{}.with(mmadParams), c1Local, l0aLocal, l0bLocal);
            }

            SetFlag<HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename C1Tensor, typename TensorBiasL1>
    __aicore__ inline void IterateABL1(
        TensorA gmA, TensorB gmB, AscendC::Te::MmadParams& mmadParams, C1Tensor& c1Local, uint64_t curML1, uint64_t curNL1,
        const TensorBiasL1& tensorBiasL1, uint16_t biasBufId)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});

        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t curKL1 = (iter0 == kL1Iter_ - 1) ? (k_ - iter0 * kL1_) : kL1_;
            uint16_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            WaitFlag<HardEvent::MTE1_MTE2>(l1BufId);

            uint64_t offsetAL1 = l1BufferAOffset_[l1BufId];
            auto layoutAL1 = MakeLayoutAL1{}(curML1, curKL1);
            auto tensorAL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAL1), layoutAL1);
            if constexpr (DispatchPolicy::fullLoadMode == 0) {
                auto gmTileA = gmA.Slice(AscendC::Te::MakeCoord(0UL, iter0 * kAL1_), AscendC::Te::MakeShape(curML1, curKL1));
                AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);
            } else {
                if (abL1LoopCnt_ < kL1Iter_) {
                    offsetAL1 = l1BufferAOffset_[0] + iter0 * kL1_ * CeilAlign(curML1, transA ? C0_SIZE : BLOCK_CUBE);
                    tensorAL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAL1), layoutAL1);
                    auto gmTileA = gmA.Slice(AscendC::Te::MakeCoord(0UL, iter0 * kL1_), AscendC::Te::MakeShape(curML1, curKL1));
                    AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);
                }
            }

            uint64_t offsetBL1 = l1BufferBOffset_[l1BufId];
            auto layoutBL1 = MakeLayoutBL1{}(curKL1, curNL1);
            auto tensorBL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(offsetBL1), layoutBL1);
            auto gmTileB = gmB.Slice(AscendC::Te::MakeCoord(iter0 * kBL1_, 0UL), AscendC::Te::MakeShape(curKL1, curNL1));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);

            SetFlag<HardEvent::MTE2_MTE1>(l1BufId);
            WaitFlag<HardEvent::MTE2_MTE1>(l1BufId);

            Iterate(
                mmadParams, tensorAL1, tensorBL1, tensorBiasL1, c1Local, curML1, curNL1, curKL1, 0UL, 0UL,
                iter0 == kL1Iter_ - 1, L0_INIT_MODE_ABL1, iter0, 0UL, biasBufId);

            SetFlag<HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }
        if constexpr (DispatchPolicy::fullLoadMode != 0) {
            abL1LoopCnt_ = 0;
        }
    }

    template <typename TensorA, typename TensorB, typename C1Tensor, typename TensorBiasL1>
    __aicore__ inline void IterateAL1BL1(
        TensorA gmA, TensorB gmB, AscendC::Te::MmadParams& mmadParams, C1Tensor& c1Local, uint64_t curML1, uint64_t curNL1,
        const TensorBiasL1& tensorBiasL1, uint16_t biasBufId)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});

        for (uint64_t kAIter = 0; kAIter < kAL1Iter_; ++kAIter) {
            uint64_t curKAL1 = (kAIter == kAL1Iter_ - 1) ? (k_ - kAIter * kAL1_) : kAL1_;
            WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + aPingPongID_);

            uint64_t offsetAL1 = l1BufferAOffset_[aPingPongID_];
            auto layoutAL1 = MakeLayoutAL1{}(curML1, curKAL1);
            auto tensorAL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAL1), layoutAL1);
            auto gmTileA = gmA.Slice(AscendC::Te::MakeCoord(0UL, kAIter * kAL1_), AscendC::Te::MakeShape(curML1, curKAL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);

            SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + aPingPongID_);
            WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + aPingPongID_);

            uint64_t kBL1IterLocal = CeilDiv(curKAL1, kBL1_);
            for (uint64_t kBIter = 0; kBIter < kBL1IterLocal; ++kBIter) {
                uint64_t curKBL1 = (kBIter == kBL1IterLocal - 1) ? (curKAL1 - kBIter * kBL1_) : kBL1_;
                WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bPingPongID_);

                uint64_t offsetBL1 = l1BufferBOffset_[bPingPongID_];
                auto layoutBL1 = MakeLayoutBL1{}(curKBL1, curNL1);
                auto tensorBL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(offsetBL1), layoutBL1);
                uint64_t bKStart = kAIter * kAL1_ + kBIter * kBL1_;
                auto gmTileB = gmB.Slice(AscendC::Te::MakeCoord(bKStart, 0UL), AscendC::Te::MakeShape(curKBL1, curNL1));
                AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);

                SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + bPingPongID_);
                WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + bPingPongID_);

                uint64_t aKPrefix = kBIter * kBL1_;
                Iterate(
                    mmadParams, tensorAL1, tensorBL1, tensorBiasL1, c1Local, curML1, curNL1, curKBL1, aKPrefix, 0UL,
                    (kAIter == kAL1Iter_ - 1) && (kBIter == kBL1IterLocal - 1), L0_INIT_MODE_SPLIT, kAIter, kBIter,
                    biasBufId);

                SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bPingPongID_);
                bPingPongID_ = bPingPongID_ ^ 1;
                abL1LoopCnt_++;
            }
            SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + aPingPongID_);
            aPingPongID_ = aPingPongID_ ^ 1;
        }
        abL1LoopCnt_ = 0;
    }

    template <typename TensorA, typename TensorB, typename C1Tensor, typename TensorBiasL1>
    __aicore__ inline void IterateBL1AL1(
        TensorA gmA, TensorB gmB, AscendC::Te::MmadParams& mmadParams, C1Tensor& c1Local, uint64_t curML1, uint64_t curNL1,
        const TensorBiasL1& tensorBiasL1, uint16_t biasBufId)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});

        for (uint64_t kBIter = 0; kBIter < kBL1Iter_; ++kBIter) {
            uint64_t curKBL1 = (kBIter == kBL1Iter_ - 1) ? (k_ - kBIter * kBL1_) : kBL1_;
            WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + bPingPongID_);

            uint64_t offsetBL1 = l1BufferBOffset_[bPingPongID_];
            auto layoutBL1 = MakeLayoutBL1{}(curKBL1, curNL1);
            auto tensorBL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(offsetBL1), layoutBL1);
            auto gmTileB = gmB.Slice(AscendC::Te::MakeCoord(kBIter * kBL1_, 0UL), AscendC::Te::MakeShape(curKBL1, curNL1));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);

            SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + bPingPongID_);
            WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + bPingPongID_);

            uint64_t kAL1IterLocal = CeilDiv(curKBL1, kAL1_);
            for (uint64_t kAIter = 0; kAIter < kAL1IterLocal; ++kAIter) {
                uint64_t curKAL1 = (kAIter == kAL1IterLocal - 1) ? (curKBL1 - kAIter * kAL1_) : kAL1_;
                WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aPingPongID_);

                uint64_t offsetAL1 = l1BufferAOffset_[aPingPongID_];
                auto layoutAL1 = MakeLayoutAL1{}(curML1, curKAL1);
                auto tensorAL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAL1), layoutAL1);
                uint64_t aKStart = kBIter * kBL1_ + kAIter * kAL1_;
                auto gmTileA = gmA.Slice(AscendC::Te::MakeCoord(0UL, aKStart), AscendC::Te::MakeShape(curML1, curKAL1));
                AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);

                SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + aPingPongID_);
                WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + aPingPongID_);

                uint64_t bKPrefix = kAIter * kAL1_;
                Iterate(
                    mmadParams, tensorAL1, tensorBL1, tensorBiasL1, c1Local, curML1, curNL1, curKAL1, 0UL, bKPrefix,
                    (kBIter == kBL1Iter_ - 1) && (kAIter == kAL1IterLocal - 1), L0_INIT_MODE_SPLIT, kBIter, kAIter,
                    biasBufId);

                SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aPingPongID_);
                aPingPongID_ = aPingPongID_ ^ 1;
                abL1LoopCnt_++;
            }
            SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + bPingPongID_);
            bPingPongID_ = bPingPongID_ ^ 1;
        }
        abL1LoopCnt_ = 0;
    }

    constexpr static uint64_t HALF_L0_SIZE = TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;

    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kAL1Iter_{0};
    uint64_t kBL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t kAL1_{1};
    uint64_t kBL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    uint16_t aPingPongID_{0};
    uint16_t bPingPongID_{0};
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t biasLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};
    uint64_t aL1OneBuffer_{0UL};
    uint64_t bL1OneBuffer_{0UL};
    uint64_t l1BufferAOffset_[4] = {0UL};
    uint64_t l1BufferBOffset_[4] = {0UL};
    uint64_t x2ScaleL1OneBuffer_{0UL};
    uint64_t biasL1OneBuffer_{0UL};
    uint64_t l1BufferX2ScaleOffset_[2] = {0UL};
    uint64_t l1BufferBiasOffset_[2] = {0UL};
    uint64_t scalarScale_{0UL};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
