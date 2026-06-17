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
 * \file kernel_qbmm_cube.h
 * \brief Quantized batch matmul cube kernel (A8W8 fixpipe, Tensor API)
 */

#pragma once

#include "kernel_universal.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_CUBE_KERNEL_TEM_PARAMS                           \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,   \
        AscendC::Std::enable_if_t<                            \
            AscendC::Std::is_same_v<KernelMmadWithScaleFixpipeQuant, typename BlockMmad::DispatchPolicy::ScheduleType>>

#define QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_CUBE_KERNEL_FUNC_TEMPLATE_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal()
    {}
    __aicore__ inline ~GemmUniversal()
    {}

    static constexpr bool weightNz = BlockMmad::weightNz;
    static constexpr bool isAtomicAdd = BlockMmad::DispatchPolicy::isAtomicAdd;
    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockMmadParams = typename BlockMmad::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using X2ScaleType = uint64_t;
    using ScaleGmType = typename BlockMmad::X2ScaleType;

    static constexpr int64_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    static constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;
    static constexpr uint32_t LEFT_SHIFT_16 = 16;

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::AuxGetC0Size<CType>()>>;

    struct QBMMTiling {
        uint32_t batchA1;
        uint32_t batchA2;
        uint32_t batchA3;
        uint32_t batchA4;
        uint32_t batchB1;
        uint32_t batchB2;
        uint32_t batchB3;
        uint32_t batchB4;
        uint32_t batchC1;
        uint32_t batchC2;
        uint32_t batchC3;
        uint32_t batchC4;
        uint32_t biasThreeDim;
        uint32_t x1QuantMode;
        uint32_t x2QuantMode;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t nBufferNum;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void AddBatchOffset(const Params& params);
    __aicore__ inline void ProcessSingleBatch(
        const Params& params, BlockScheduler& bs, uint64_t batchCnt, bool isTailRound);
    __aicore__ inline void ProcessWithBatch(const Params& params, BlockScheduler& bs);

    BlockMmad mmadOp_;
    __gm__ AType* aGmBase_{nullptr};
    __gm__ BType* bGmBase_{nullptr};
    __gm__ CType* cGmBase_{nullptr};
    __gm__ BiasType* biasGmBase_{nullptr};
    __gm__ X2ScaleType* scaleGmBase_{nullptr};
    bool isBias_{false};
    bool isBiasThreeDim_{false};
    uint64_t scaleScalar_{0};
    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    bool needUpdateTail_{false};
};

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::Run(const Params& params)
{
    if constexpr (isAtomicAdd) {
        AscendC::SetAtomicAdd<float>();
    }
    Init(params);
    BlockScheduler bs(params.problemShape, params.schParams);

    BlockShape l0TileShape{
        static_cast<int64_t>(params.qbmmParams.baseM), static_cast<int64_t>(params.qbmmParams.baseN),
        static_cast<int64_t>(params.qbmmParams.baseK), 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
    mmadOp_.Init(
        params.problemShape, l0TileShape, params.qbmmParams.kAL1, params.qbmmParams.kBL1, params.qbmmParams.nBufferNum,
        static_cast<QuantMode>(params.qbmmParams.x2QuantMode), isBias_, enableL0CPingPong);

    if (AscendC::Te::Get<MNK_B>(params.problemShape) == 1) {
        AddBatchOffset(params);
        ProcessSingleBatch(params, bs, 0, true);
        if constexpr (isAtomicAdd) {
            AscendC::SetAtomicNone();
        }
        return;
    }

    ProcessWithBatch(params, bs);
    if constexpr (isAtomicAdd) {
        AscendC::SetAtomicNone();
    }
}

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (params.qbmmParams.isBias == 1) {
        isBias_ = true;
        biasGmBase_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
        if (params.qbmmParams.biasThreeDim == 1) {
            isBiasThreeDim_ = true;
        }
    }
    aGmBase_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmBase_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmBase_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    if (static_cast<QuantMode>(params.qbmmParams.x2QuantMode) == QuantMode::PERCHANNEL_MODE) {
        scaleGmBase_ = reinterpret_cast<__gm__ uint64_t*>(params.mmadParams.scaleBGmAddr);
    } else if (
        static_cast<QuantMode>(params.qbmmParams.x1QuantMode) == QuantMode::PERTENSOR_MODE) {
        auto pertokenScale = AscendC::GlobalTensor<float>();
        auto scale = AscendC::GlobalTensor<float>();
        pertokenScale.SetGlobalBuffer((__gm__ float*)params.mmadParams.scaleAGmAddr);
        scale.SetGlobalBuffer((__gm__ float*)params.mmadParams.scaleBGmAddr);
        float deqScale = pertokenScale.GetValue(0) * scale.GetValue(0);
        uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&deqScale));
        scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
    } else if (static_cast<QuantMode>(params.qbmmParams.x2QuantMode) == QuantMode::PERTENSOR_MODE) {
        if constexpr (AscendC::IsSameType<ScaleGmType, uint64_t>::value ||
                      AscendC::IsSameType<ScaleGmType, int64_t>::value) {
            // Use GlobalTensor GM load (same as CMCT); do not dereference GM_ADDR directly on AICore.
            auto scale = AscendC::GlobalTensor<uint64_t>();
            scale.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(params.mmadParams.scaleBGmAddr));
            scaleScalar_ = scale.GetValue(0);
        } else if constexpr (AscendC::IsSameType<ScaleGmType, bfloat16_t>::value) {
            auto scale = AscendC::GlobalTensor<uint16_t>();
            scale.SetGlobalBuffer((__gm__ uint16_t*)params.mmadParams.scaleBGmAddr);
            uint16_t uint16Scale = scale.GetValue(0);
            uint32_t uint32Scale = static_cast<uint32_t>(uint16Scale << LEFT_SHIFT_16);
            scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
        } else {
            auto scale = AscendC::GlobalTensor<uint32_t>();
            scale.SetGlobalBuffer((__gm__ uint32_t*)params.mmadParams.scaleBGmAddr);
            uint32_t uint32Scale = scale.GetValue(0);
            scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
        }
    }
}

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmBase_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmBase_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmBase_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    if (isBias_) {
        biasGmBase_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }
}

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::AddBatchOffset(const Params& params)
{
    ResetGmAddr(params);
    aGmBase_ += batchAOffset_ * AscendC::Te::Get<MNK_M>(params.problemShape) *
                AscendC::Te::Get<MNK_K>(params.problemShape);
    if constexpr (weightNz) {
        if constexpr (transB) {
            bGmBase_ += batchBOffset_ * Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_K>(params.problemShape), C0_SIZE) *
                        Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_N>(params.problemShape), static_cast<int64_t>(BLOCK_CUBE)) *
                        BLOCK_CUBE * C0_SIZE;
        } else {
            bGmBase_ += batchBOffset_ * Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_N>(params.problemShape), C0_SIZE) *
                        Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(BLOCK_CUBE)) *
                        BLOCK_CUBE * C0_SIZE;
        }
    } else {
        bGmBase_ += batchBOffset_ * AscendC::Te::Get<MNK_N>(params.problemShape) *
                    AscendC::Te::Get<MNK_K>(params.problemShape);
    }
    cGmBase_ += batchCOffset_ * AscendC::Te::Get<MNK_M>(params.problemShape) *
                AscendC::Te::Get<MNK_N>(params.problemShape);
    if (isBiasThreeDim_) {
        biasGmBase_ += batchCOffset_ * AscendC::Te::Get<MNK_N>(params.problemShape);
    }
}

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::ProcessWithBatch(
    const Params& params, BlockScheduler& bs)
{
    uint64_t batchC3C4 = static_cast<uint64_t>(params.qbmmParams.batchC3) * params.qbmmParams.batchC4;
    uint64_t batchC2C3C4 = params.qbmmParams.batchC2 * batchC3C4;
    uint64_t batchB3B4 = static_cast<uint64_t>(params.qbmmParams.batchB3) * params.qbmmParams.batchB4;
    uint64_t batchB2B3B4 = params.qbmmParams.batchB2 * batchB3B4;
    uint64_t batchA3A4 = static_cast<uint64_t>(params.qbmmParams.batchA3) * params.qbmmParams.batchA4;
    uint64_t batchA2A3A4 = params.qbmmParams.batchA2 * batchA3A4;
    uint32_t multiA1C1 = params.qbmmParams.batchA1 / params.qbmmParams.batchC1;
    uint32_t multiA2C2 = params.qbmmParams.batchA2 / params.qbmmParams.batchC2;
    uint32_t multiA3C3 = params.qbmmParams.batchA3 / params.qbmmParams.batchC3;
    uint32_t multiA4C4 = params.qbmmParams.batchA4 / params.qbmmParams.batchC4;
    uint32_t multiB1C1 = params.qbmmParams.batchB1 / params.qbmmParams.batchC1;
    uint32_t multiB2C2 = params.qbmmParams.batchB2 / params.qbmmParams.batchC2;
    uint32_t multiB3C3 = params.qbmmParams.batchB3 / params.qbmmParams.batchC3;
    uint32_t multiB4C4 = params.qbmmParams.batchB4 / params.qbmmParams.batchC4;

    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    uint64_t curBatchC = 1UL;
    uint64_t totalCnt = bs.GetTotalCnt() * AscendC::Te::Get<MNK_B>(params.problemShape);
    uint64_t nonTailRoundCnt = (totalCnt / AscendC::GetBlockNum()) * AscendC::GetBlockNum();
    for (uint64_t b1Index = 0; b1Index < params.qbmmParams.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;
        for (uint64_t b2Index = 0; b2Index < params.qbmmParams.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;
            for (uint64_t b3Index = 0; b3Index < params.qbmmParams.batchC3; ++b3Index) {
                batchCOffset_ = batchC3Offset;
                batchAOffset_ = batchA3Offset;
                batchBOffset_ = batchB3Offset;
                for (uint64_t b4Index = 0; b4Index < params.qbmmParams.batchC4; ++b4Index) {
                    bool isTailRound = curBatchC * bs.GetTotalCnt() > nonTailRoundCnt;
                    AddBatchOffset(params);
                    ProcessSingleBatch(
                        params, bs, (AscendC::Te::Get<MNK_B>(params.problemShape) - curBatchC), isTailRound);
                    curBatchC++;
                    batchCOffset_ += 1;
                    batchAOffset_ += multiA4C4;
                    batchBOffset_ += multiB4C4;
                }
                batchC3Offset += params.qbmmParams.batchC4;
                batchA3Offset += params.qbmmParams.batchA4 * static_cast<uint64_t>(multiA3C3);
                batchB3Offset += params.qbmmParams.batchB4 * static_cast<uint64_t>(multiB3C3);
            }
            batchC2Offset += batchC3C4;
            batchA2Offset += batchA3A4 * multiA2C2;
            batchB2Offset += batchB3B4 * multiB2C2;
        }
        batchC1Offset += batchC2C3C4;
        batchA1Offset += batchA2A3A4 * multiA1C1;
        batchB1Offset += batchB2B3B4 * multiB1C1;
    }
}

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::ProcessSingleBatch(
    const Params& params, BlockScheduler& bs, uint64_t restBatch, bool isTailRound)
{
    const int64_t m = AscendC::Te::Get<MNK_M>(params.problemShape);
    const int64_t n = AscendC::Te::Get<MNK_N>(params.problemShape);
    const int64_t k = AscendC::Te::Get<MNK_K>(params.problemShape);
    constexpr int64_t kPos = 0;

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutC = MakeLayoutC{}(m, n);

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmBase_), layoutA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmBase_), layoutB);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmBase_), layoutC);

    auto layoutBias = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1L, n);
    __gm__ BiasType* biasPtr = isBias_ ? biasGmBase_ : reinterpret_cast<__gm__ BiasType*>(cGmBase_);
    auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasPtr), layoutBias);

    const bool isPerChannel =
        static_cast<QuantMode>(params.qbmmParams.x2QuantMode) == QuantMode::PERCHANNEL_MODE;

    BlockCoord blockIdx;
    if (needUpdateTail_ || (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) *
                                                   params.schParams.mTailTile * params.schParams.nTailTile <=
                                               AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(params.schParams.mTailTile, params.schParams.nTailTile);
    }

    int64_t mPos = 0L;
    int64_t nPos = 0L;
    auto layoutScale = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<X2ScaleType>>(1, n);
    auto gmScale = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleGmBase_), layoutScale);
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape = bs.template GetBlockShape<QuantMode::DEFAULT, QuantMode::DEFAULT, weightNz>(blockIdx);
        if (AscendC::Te::Get<IDX_M_TILEIDX>(singleShape) <= 0 || AscendC::Te::Get<IDX_N_TILEIDX>(singleShape) <= 0) {
            return;
        }

        bs.GetTileCoord(blockIdx, mPos, nPos);
        const int64_t curM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        const int64_t curN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);

        auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(curM, k));
        auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(k, curN));
        auto gmBlockC = gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(curM, curN));

        if (isPerChannel) {
            auto gmBlockScale = gmScale.Slice(AscendC::Te::MakeCoord(0UL, nPos), AscendC::Te::MakeShape(1, curN));
            if (isBias_) {
                auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(0, nPos), AscendC::Te::MakeShape(1, curN));
                mmadOp_(gmBlockA, gmBlockB, gmBlockScale, gmBlockBias, gmBlockC, singleShape);
            } else {
                auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(0, 0), AscendC::Te::MakeShape(1, 1));
                mmadOp_(gmBlockA, gmBlockB, gmBlockScale, gmBlockBias, gmBlockC, singleShape);
            }
        } else {
            if (isBias_) {
                auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(0, nPos), AscendC::Te::MakeShape(1, curN));
                mmadOp_(gmBlockA, gmBlockB, scaleScalar_, gmBlockBias, gmBlockC, singleShape);
            } else {
                auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(0, 0), AscendC::Te::MakeShape(1, 1));
                mmadOp_(gmBlockA, gmBlockB, scaleScalar_, gmBlockBias, gmBlockC, singleShape);
            }
        }
    }
    bs.UpdateNextBatchBlockRoundParams();
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
