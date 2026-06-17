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
 * \file block_mmad_matmul_basic.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <
    uint64_t FULL_LOAD_MODE_, uint64_t FUSED_OP_TYPE_, class KernelSchedule_, class AType_, class LayoutA_, class BType_, class LayoutB_,
    class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<
    MatmulMultiBlockBasic<FULL_LOAD_MODE_, FUSED_OP_TYPE_, KernelSchedule_>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
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
    using DispatchPolicy = MatmulMultiBlockBasic<FULL_LOAD_MODE_, FUSED_OP_TYPE_, KernelSchedule_>;
    using TupleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};

    constexpr static uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(AType);
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
    constexpr static uint64_t HALF_L1_SIZE = AscendC::TOTAL_L1_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t QUARTER_L1_SIZE = AscendC::TOTAL_L1_SIZE / QUADRUPLE_BUFFER_COUNT;
    constexpr static uint16_t MTE1_MTE2_EVENT_ID_NUM = 4;

    // transA and transB
    static constexpr bool transA = IsTrans<LayoutA>::value;
    static constexpr bool transB = IsTrans<LayoutB>::value;
    static constexpr bool weightNZFormat = IsWeightNz<LayoutB>::value;
    // AL1 Layout
    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>>;
    // BL1 Layout
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>>;

    // host side kernel arguments
    struct Arguments {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
    };

    // params
    using Params = Arguments;

private:
    uint64_t kL1Iter_{0};
    uint64_t l1BufNum_{1};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool isBias_{false};
    bool enableL0cPingPong_{false};

public:
    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
        }
    }

    __aicore__ inline void Init(
        const TupleShape& shape, const TupleShape& tileL1, const TupleShape& tileL0, bool isBias, uint64_t l1BufNum,
        bool l0cDB)
    {
        m_ = AscendC::Te::Get<DIMENSION_M>(shape);
        n_ = AscendC::Te::Get<DIMENSION_N>(shape);
        k_ = AscendC::Te::Get<DIMENSION_K>(shape);
        mL1_ = AscendC::Te::Get<DIMENSION_M>(tileL1);
        nL1_ = AscendC::Te::Get<DIMENSION_N>(tileL1);
        kL1_ = AscendC::Te::Get<DIMENSION_K>(tileL1);
        baseM_ = AscendC::Te::Get<DIMENSION_M>(tileL0);
        baseN_ = AscendC::Te::Get<DIMENSION_N>(tileL0);
        baseK_ = AscendC::Te::Get<DIMENSION_K>(tileL0);
        isBias_ = isBias;
        l1BufNum_ = l1BufNum;
        enableL0cPingPong_ = l0cDB;
        // 非全载
        aL1OneBuffer_ = mL1_ * kL1_ * sizeof(AType);
        bL1OneBuffer_ = nL1_ * kL1_ * sizeof(BType);
        kL1Iter_ = CeilDiv(k_, kL1_);
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
    }

    template <typename TensorC, typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline void operator()(
        TensorC gmC, TensorA gmA, TensorB gmB, TensorBias gmBias, TupleL1L0Shape tileShape)
    {
        // m0 n0
        uint64_t curM = AscendC::Te::Get<MNK_M0>(tileShape);
        uint64_t curN = AscendC::Te::Get<MNK_N0>(tileShape);
        uint64_t ml1Align = Blaze::Gemm::CeilAlign(curM, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        uint64_t nl1Align = Blaze::Gemm::CeilAlign(curN, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        uint64_t l0cOffset = (l0cPingPong_ & 0x1) * HALF_L0C_SIZE;
        if (enableL0cPingPong_) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
        }
        kL1_ = Min(k_, kL1_);

        // LoC搬出
        auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<16>>{}(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset * sizeof(float)), layoutL0C);

        kL1Iter_ = CeilDiv(k_, kL1_);
        uint64_t kL1OffsetLength = 0;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            auto curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1OffsetLength) : kL1_;
            // 普通模板-2buffer-AL1搬入偏移位置：*AL1Ping*-BL1Ping-BiasPing|*AL1Pong*-BL1Pong-BiasPong
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

            // A GM->L1
            auto layoutAL1 = MakeLayoutAL1{}(curM, curKL1);
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            uint64_t offsetAl1 =
                (l1BufNum_ == DOUBLE_BUFFER_COUNT) ? HALF_L1_SIZE * l1BufId : QUARTER_L1_SIZE * l1BufId;
            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAl1), layoutAL1);
            auto gmTileA = gmA.Slice(AscendC::Te::MakeCoord(0, iter0 * kL1_), AscendC::Te::MakeShape(curM, curKL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);

            // Bias GM->L1
            uint64_t biasBufId = abL1LoopCnt_ & 0x1;
            uint64_t offsetBiasL1 = (l1BufNum_ == DOUBLE_BUFFER_COUNT) ?
                                        HALF_L1_SIZE * l1BufId + aL1OneBuffer_ + bL1OneBuffer_ :
                                        QUARTER_L1_SIZE * l1BufId + aL1OneBuffer_ + bL1OneBuffer_;
            auto layoutBiasL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curN);
            auto tensorBiasL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(offsetBiasL1), layoutBiasL1);
            // 非全载
            if (isBias_ && iter0 == 0) {
                AscendC::Te::Copy(copyGM2L1, tensorBiasL1, gmBias);
            }

            // B GM->L1
            auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
            uint64_t offsetBl1 = (l1BufNum_ == DOUBLE_BUFFER_COUNT) ? HALF_L1_SIZE * l1BufId + aL1OneBuffer_ :
                                                                      QUARTER_L1_SIZE * l1BufId + aL1OneBuffer_;
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(offsetBl1), layoutBL1);
            auto gmTileB = gmB.Slice(AscendC::Te::MakeCoord(iter0 * kL1_, 0), AscendC::Te::MakeShape(curKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            kL1OffsetLength += curKL1;
            uint64_t kL0Iter = (curKL1 + baseK_ - 1) / baseK_;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                uint64_t mte1Flag = ((l0PingPong_ & 0x1) + SIXTH_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));

                // A L1->L0
                auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
                auto layoutAL0 =
                    AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(
                        curM, curK0);
                auto tensorAL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset * sizeof(AType)), layoutAL0);
                auto tensorBlockAL1 =
                    tensorAL1.Slice(AscendC::Te::MakeCoord(0, iter1 * baseK_), AscendC::Te::MakeShape(curM, curK0));
                AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);

                // Bias L1->L0
                uint64_t nl1Align = Blaze::Gemm::CeilAlign(curN, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
                auto layoutBiasL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, nl1Align);
                auto offsetBiasL0 = baseN_ * biasBufId * sizeof(float);
                auto tensorBiasL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, float>(offsetBiasL0), layoutBiasL0);
                if (NeedProcessBias(iter0, iter1)) {
                    auto copyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
                    AscendC::Te::Copy(copyL12BT, tensorBiasL0, tensorBiasL1);
                }

                // B L1->L0
                auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
                auto layoutBL0 =
                    AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(
                        curK0, curN);
                auto tensorBL0 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset * sizeof(BType)), layoutBL0);
                auto tensorBlockBL1 =
                    tensorBL1.Slice(AscendC::Te::MakeCoord(iter1 * baseK_, 0), AscendC::Te::MakeShape(curK0, curN));
                AscendC::Te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));

                constexpr auto mmadAtom =
                    AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});

                // Mmad参数
                AscendC::Te::MmadParams mmadParams(
                    curM, curN, curK0,
                    (enableL0cPingPong_ ? 0 :
                                          ((iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION :
                                                                                             NON_FINAL_ACCUMULATION)),
                    (iter0 == 0 && iter1 == 0 && !isBias_));
                // 传入自定义Trait类型
                if (NeedProcessBias(iter0, iter1)) {
                    AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
                } else {
                    AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                l0PingPong_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }
        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
        }

        // 数据搬出到GM
        AscendC::Te::FixpipeParams fixpParams(enableL0cPingPong_ ? 0 : FINAL_ACCUMULATION);
        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(copyL0C2GM.with(fixpParams), gmC, tensorL0C);

        if (enableL0cPingPong_) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
            l0cPingPong_++;
        }
    }

private:
    __aicore__ inline bool NeedProcessBias(uint64_t kIter0, uint64_t kIter1)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0;
    }

private:
    constexpr static uint16_t DIMENSION_M = 0;
    constexpr static uint16_t DIMENSION_N = 1;
    constexpr static uint16_t DIMENSION_K = 2;
    constexpr static uint16_t ZERO_FLAG = 0;
    constexpr static uint16_t FIRST_FLAG = 1;
    constexpr static uint16_t SECOND_FLAG = 2;
    constexpr static uint16_t THIRD_FLAG = 3;
    constexpr static uint16_t FOURTH_FLAG = 4;
    constexpr static uint16_t FIFTH_FLAG = 5;
    constexpr static uint16_t SIXTH_FLAG = 6;
    constexpr static uint16_t SEVENTH_FLAG = 7;
    constexpr static int32_t BT_SIZE = 4096;
    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze