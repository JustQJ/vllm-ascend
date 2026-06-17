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
 * \file block_mmad_matmul_streamk.h
 * \brief
 */

#pragma once

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"
#include "block_mmad.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <
    MatMulL0C2Out FixpOpti_, uint64_t FUSED_OP_TYPE_, class AType_, class LayoutA_, class BType_, class LayoutB_,
    class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<
    MatmulMultiBlockWithStreamK<FixpOpti_, FUSED_OP_TYPE_>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
    BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = MatmulMultiBlockWithStreamK<FixpOpti_, FUSED_OP_TYPE_>;
    using TupleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = !(AscendC::Std::is_same_v<LayoutA, AscendC::Te::NDExtLayoutPtn>);
    static constexpr bool transB =
        !(AscendC::Std::is_same_v<LayoutB, AscendC::Te::NDExtLayoutPtn> ||
          AscendC::Std::is_same_v<LayoutB, AscendC::Te::NZLayoutPtn>);
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};

    bool isBias_{false};
    constexpr static uint64_t BUFFER_NUM = 2;
    constexpr static uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / BUFFER_NUM;
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    struct GmParams {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
    };

private:
    uint64_t biasL1Offset_ = 0;
    uint64_t bL1Init_ = 0;
    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
    constexpr static uint16_t L1_EVENT_ID_OFFSET = 2;
    constexpr static uint16_t MTE1_MTE2_EVENT_ID_NUM = 4;

public:
    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>>;

    __aicore__ inline BlockMmad()
    {
        for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    }

    __aicore__ inline ~BlockMmad()
    {
        for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    }

public:
    __aicore__ inline void Init(
        const TupleShape& shape, const TupleShape& tileL1, const TupleShape& tileL0, bool isBias)
    {
        m_ = Get<MNK_M>(shape);
        n_ = Get<MNK_N>(shape);
        k_ = Get<MNK_K>(shape);

        mL1_ = Get<MNK_M>(tileL1);
        nL1_ = Get<MNK_N>(tileL1);
        kL1_ = Get<MNK_K>(tileL1);

        baseM_ = Get<MNK_M>(tileL0);
        baseN_ = Get<MNK_N>(tileL0);
        baseK_ = Get<MNK_K>(tileL0);
        isBias_ = isBias;
        // init tensor
        if (isBias_) {
            biasL1Offset_ = nL1_ * sizeof(BiasType) * BUFFER_NUM;
        }
        aL1OneBuffer_ = mL1_ * kL1_;
        bL1Init_ = biasL1Offset_ + aL1OneBuffer_ * BUFFER_NUM;
        bL1OneBuffer_ = nL1_ * kL1_;
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
    }

    template <typename TensorC, typename TensorA, typename TensorB, typename TensorBias, typename TensorWorkspace>
    __aicore__ inline void operator()(
        TensorC gmC, TensorA gmA, TensorB gmB, TensorBias gmBias, TensorWorkspace gmWorkspace, TupleShape tileShape,
        int64_t kCntIndex, bool checkIsSkScene)
    {
        // mL1_ == ml0, nL1_ == nl0
        uint64_t curML1 = Get<MNK_M>(tileShape);
        uint64_t curNL1 = Get<MNK_N>(tileShape);
        uint64_t curSingleCoreK = Get<MNK_K>(tileShape);
        uint64_t curKL1Iter = (curSingleCoreK + kL1_ - 1) / kL1_;
        uint64_t nl1Align = CeilAlign(curNL1, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        uint64_t l0cOffset = 0;
        auto layoutL0C =
            AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>{}(curML1, curNL1);
        auto tensorL0C =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset), layoutL0C);
        for (uint64_t iter0 = 0; iter0 < curKL1Iter; ++iter0) {
            uint64_t curKL1 = (iter0 + 1 == curKL1Iter) ? (curSingleCoreK - iter0 * kL1_) : kL1_;
            // switch on pingpong, now only support double buffer in streamk
            uint64_t l1BufId = abL1LoopCnt_ & (BUFFER_NUM - 1);
            uint64_t offsetAL1 = (biasL1Offset_ + aL1OneBuffer_ * l1BufId) * sizeof(AType);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1());

            // copy bias to l1
            uint64_t biasBufId = abL1LoopCnt_ & 0x1;
            uint64_t offsetBiasL1 = nL1_ * l1BufId * sizeof(BiasType);
            auto layoutBiasL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curNL1);
            auto tensorBiasL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(offsetBiasL1), layoutBiasL1);
            if (isBias_ && iter0 == 0 && kCntIndex == 0) {
                AscendC::Te::Copy(copyGM2L1, tensorBiasL1, gmBias);
            }
            // copy tensor a to l1
            auto layoutAL1 = MakeLayoutAL1{}(curML1, curKL1);
            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAL1), layoutAL1);
            auto gmTileA = gmA.Slice(AscendC::Te::MakeCoord(0, iter0 * kL1_), AscendC::Te::MakeShape(curML1, curKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            // copy tensor b to l1
            uint64_t offsetBL1 = (bL1Init_ + bL1OneBuffer_ * l1BufId) * sizeof(BType);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId + L1_EVENT_ID_OFFSET);
            auto layoutBL1 = MakeLayoutBL1{}(curKL1, curNL1);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(offsetBL1), layoutBL1);
            auto gmTileB = gmB.Slice(AscendC::Te::MakeCoord(iter0 * kL1_, 0), AscendC::Te::MakeShape(curKL1, curNL1));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId + L1_EVENT_ID_OFFSET);

            uint64_t kL0Iter = (curKL1 + baseK_ - 1) / baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                // copy aL1 to l0a
                auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                auto layoutAL0 =
                    AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(
                        curML1, curK0);
                auto tensorAL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
                auto tensorBlockAL1 =
                    tensorAL1.Slice(AscendC::Te::MakeCoord(0, iter1 * baseK_), AscendC::Te::MakeShape(curML1, curK0));
                AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);
                if (iter1 == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId + L1_EVENT_ID_OFFSET);
                }
                // copy bias to biastable
                auto layoutBiasL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, nl1Align);
                uint64_t offsetBiasL0 = nL1_ * biasBufId * sizeof(float);
                auto tensorBiasL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, float>(offsetBiasL0), layoutBiasL0);
                if (isBias_ && iter0 == 0 && iter1 == 0 && kCntIndex == 0) {
                    auto copyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
                    AscendC::Te::Copy(copyL12BT, tensorBiasL0, tensorBiasL1);
                }
                // copy bL1 to l0b
                auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
                auto layoutBL0 =
                    AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(
                        curK0, curNL1);
                auto tensorBL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);
                auto tensorBlockBL1 =
                    tensorBL1.Slice(AscendC::Te::MakeCoord(iter1 * baseK_, 0), AscendC::Te::MakeShape(curK0, curNL1));
                AscendC::Te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                uint8_t unitFlag =
                    (iter0 + 1 == curKL1Iter && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
                bool cmatrixInitVal = (iter0 == 0 && iter1 == 0 && (!isBias_ || (isBias_ && kCntIndex != 0)));
                AscendC::Te::MmadParams mmadParams(curML1, curNL1, curK0, unitFlag, cmatrixInitVal);
                constexpr auto mmadAtom =
                    AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});

                if (isBias_ && iter0 == 0 && iter1 == 0 && kCntIndex == 0) {
                    AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
                } else {
                    AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                l0PingPong_++;
            }
            if (iter0 + 1 == curKL1Iter) {
                auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
                if (checkIsSkScene) {
                    AscendC::Te::Copy(
                        CopyL0C2GM.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), gmWorkspace, tensorL0C);
                } else {
                    AscendC::Te::Copy(CopyL0C2GM.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), gmC, tensorL0C);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId + L1_EVENT_ID_OFFSET);
            abL1LoopCnt_++;
        }
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze