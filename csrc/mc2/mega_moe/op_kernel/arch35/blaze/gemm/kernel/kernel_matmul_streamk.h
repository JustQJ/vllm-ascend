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
 * \file kernel_matmul_streamk.h
 * \brief
 */

#pragma once

#include "kernel_universal.h"

#define ASCENDC_CUBE_ONLY
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_matmul_streamk.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<KernelMultiBlockStreamK, typename BlockMmad_::DispatchPolicy::ScheduleType> &&
        AscendC::Std::is_same_v<KernelMultiBlockStreamK, typename BlockEpilogue_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal()
    {}
    __aicore__ inline ~GemmUniversal()
    {}
    using BlockMmadOp = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    // mmadOp
    using BlockMmadParams = typename BlockMmadOp::GmParams;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using AType = typename BlockMmadOp::AType;
    using BType = typename BlockMmadOp::BType;
    using CType = typename BlockMmadOp::CType;
    using BiasType = typename BlockMmadOp::BiasType;
    using LayoutA = typename BlockMmadOp::LayoutA;
    using LayoutB = typename BlockMmadOp::LayoutB;
    using LayoutC = typename BlockMmadOp::LayoutC;
    using LayoutBias = typename BlockMmadOp::LayoutBias;
    using TupleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_;
    __gm__ float* workspaceGmAddr_;

    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Te::LayoutTraitDefault<AType>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Te::LayoutTraitDefault<BType>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Te::LayoutTraitDefault<CType>>;
    using MakeLayoutBias = AscendC::Te::FrameLayoutFormat<LayoutBias, AscendC::Te::LayoutTraitDefault<BiasType>>;

    // basic args
    int64_t m_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t usedCoreNum_ = 0;
    // shape
    TupleShape problemShape_{};
    bool isBias_ = false;

    constexpr static uint16_t NUM_TWO = 2;
    constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static uint16_t FLAG_ID_MAX = 16;
    constexpr static uint16_t BLOCK_BASE_M = 256;
    constexpr static uint16_t BLOCK_BASE_N = 256;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = params.problemShape;
        BlockMmadParams blockMmadParams_ = params.mmadParams;
        BlockEpilogueParams blockEpilogueParams_ = params.epilogueParams;
        m_ = Get<MNK_M>(problemShape_);
        n_ = Get<MNK_N>(problemShape_);
        k_ = Get<MNK_K>(problemShape_);
        usedCoreNum_ = params.schParams.usedCoreNum;
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams_.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams_.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams_.cGmAddr);
        workspaceGmAddr_ = reinterpret_cast<__gm__ float*>(blockMmadParams_.workspaceGmAddr);
        if (blockMmadParams_.biasGmAddr != nullptr) {
            isBias_ = true;
            biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams_.biasGmAddr);
        }
    }

    __aicore__ inline void operator()(Params const& params)
    {
        Init(params);
        if (usedCoreNum_ <= 0) {
            return;
        }
        BlockScheduler bs(params.problemShape, params.schParams);
        TupleShape tileL1 = bs.GetTileL1Shape();
        int64_t mL1 = Get<MNK_M>(tileL1);
        int64_t nL1 = Get<MNK_N>(tileL1);
        int64_t kL1 = Get<MNK_K>(tileL1);
        int64_t mTileNum = Get<MNK_M>(bs.GetMNKTileNum());
        int64_t nTileNum = Get<MNK_N>(bs.GetMNKTileNum());
        int64_t skKTileNum = Get<MNK_K>(bs.GetMNKTileNum()); // it only used in sk scene
        int64_t tileNum = bs.GetTotalTileNum();
        if ASCEND_IS_AIC {
            // Instantiate mmadOp
            BlockMmadOp blockMmadOp;
            int64_t curBlockIdx = AscendC::GetBlockIdx();

            TupleShape tileL0 = bs.GetTileL0Shape();
            int64_t isHf32 = bs.GetHf32Flag();

            if (curBlockIdx >= bs.GetBlockNum(usedCoreNum_)) {
                CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
                CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
                return;
            }
            if (isHf32) {
                AscendC::SetHF32Mode(1);
                AscendC::SetHF32TransMode(1);
            }
            SetMMLayoutTransform(true); // copy out with nfirst, try to make cube and fixp pairing.
            blockMmadOp.Init(problemShape_, tileL1, tileL0, isBias_);
            int64_t tailSKTotalTileNum = static_cast<int64_t>(((mTileNum * nTileNum) % usedCoreNum_) * skKTileNum);
            auto layoutA = MakeLayoutA{}(m_, k_);
            auto layoutB = MakeLayoutB{}(k_, n_);
            auto layoutC = MakeLayoutC{}(m_, n_);
            auto layoutBias = MakeLayoutBias{}(1L, n_);
            auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
            auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
            auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);
            auto gmBias =
                AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_), layoutBias);
            for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += usedCoreNum_) {
                int64_t tmpTileIdx = tileIdx;
                if (!bs.CheckIsSkScene(0)) { // SK Preload in DP+SK
                    if (tileIdx % usedCoreNum_ < tailSKTotalTileNum &&
                        (CeilDiv(tileIdx + 1, usedCoreNum_) == (CeilDiv(tileNum, usedCoreNum_) - 1))) {
                        tmpTileIdx = tileIdx + usedCoreNum_;
                    } else if (
                        tileIdx % usedCoreNum_ < tailSKTotalTileNum &&
                        (CeilDiv(tileIdx + 1, usedCoreNum_) == CeilDiv(tileNum, usedCoreNum_))) {
                        tmpTileIdx = tileIdx - usedCoreNum_;
                    }
                }
                auto singleCoreShape = bs.GetSingleCoreShape(tmpTileIdx);
                auto singleCoreCoord = bs.GetSingleCoreCoord(tmpTileIdx);
                int64_t kSingleCore = bs.GetCurKSingleCore(tmpTileIdx);
                int64_t offsetWorkspace =
                    (((tmpTileIdx % usedCoreNum_) / skKTileNum) * skKTileNum + Get<MNK_K>(singleCoreCoord)) *
                    BLOCK_BASE_M * BLOCK_BASE_N;
                // when fixpipe 1v2, dstStride should align to 32
                auto workspaceStrideColumn0 = BlockMmadOp::DispatchPolicy::fixpOpti_ == MatMulL0C2Out::ND_FIXPIPE_1_2 ?
                                                  CeilAlign(Get<MNK_N>(singleCoreShape), BLOCK_BYTE_SIZE) :
                                                  Get<MNK_N>(singleCoreShape);
                auto workspaceShape = AscendC::Te::MakeShape(
                    AscendC::Te::MakeShape(Std::Int<1>{}, Get<MNK_M>(singleCoreShape)),
                    AscendC::Te::MakeShape(Std::Int<1>{}, Get<MNK_N>(singleCoreShape)));
                auto workspaceStride = AscendC::Te::MakeStride(
                    AscendC::Te::MakeStride(Std::Int<0>{}, workspaceStrideColumn0),
                    AscendC::Te::MakeStride(Std::Int<0>{}, Std::Int<1>{}));
                auto layoutWorkspace =
                    AscendC::Te::MakePatternLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>(
                        workspaceShape, workspaceStride);
                // workspace use 1 dim expression, make tensor each calculate
                auto gmWorkSpace = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(workspaceGmAddr_ + offsetWorkspace),
                    layoutWorkspace);
                // split tensor from gm which needed by current calculate
                auto gmBlockA = gmA.Slice(
                    AscendC::Te::MakeCoord(
                        Get<MNK_M>(singleCoreCoord) * mL1,
                        Get<MNK_K>(singleCoreCoord) * kSingleCore), // use point address, not tile block
                    AscendC::Te::MakeShape(Get<MNK_M>(singleCoreShape), Get<MNK_K>(singleCoreShape)));
                auto gmBlockB = gmB.Slice(
                    AscendC::Te::MakeCoord(
                        Get<MNK_K>(singleCoreCoord) * kSingleCore, Get<MNK_N>(singleCoreCoord) * nL1),
                    AscendC::Te::MakeShape(Get<MNK_K>(singleCoreShape), Get<MNK_N>(singleCoreShape)));
                auto gmBlockC = gmC.Slice(
                    AscendC::Te::MakeCoord(Get<MNK_M>(singleCoreCoord) * mL1, Get<MNK_N>(singleCoreCoord) * nL1),
                    AscendC::Te::MakeShape(Get<MNK_M>(singleCoreShape), Get<MNK_N>(singleCoreShape)));
                auto gmBlockBias = gmBias.Slice(
                    AscendC::Te::MakeCoord(0L, Get<MNK_N>(singleCoreCoord) * nL1),
                    AscendC::Te::MakeShape(1L, Get<MNK_N>(singleCoreShape)));
                blockMmadOp(
                    gmBlockC, gmBlockA, gmBlockB, gmBlockBias, gmWorkSpace, singleCoreShape,
                    Get<MNK_K>(singleCoreCoord), bs.CheckIsSkScene(tmpTileIdx));
                if (tmpTileIdx + usedCoreNum_ >= tileNum) {
                    CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
                    CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
                }
            }
            SetMMLayoutTransform(false);
            if (isHf32) {
                AscendC::SetHF32Mode(0);
            }
        }

        if ASCEND_IS_AIV {
            uint64_t lastLoopTotalCnt = (mTileNum * nTileNum % usedCoreNum_) * skKTileNum;
            uint64_t curBlockIdxInAiv = AscendC::GetBlockIdx();
            if (curBlockIdxInAiv >= lastLoopTotalCnt * AscendC::GetTaskRation()) {
                CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
                SyncAll();
                return;
            }

            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
            SyncAll();
            BlockEpilogue epilogueOp;
            epilogueOp.Init(
                params.epilogueParams, problemShape_, tileL1, bs.GetMNKTileNum(), usedCoreNum_, bs.CheckIsSkScene(0));
            epilogueOp();
        }
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze