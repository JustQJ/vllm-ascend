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
 * \file kernel_qbmm_mx.h
 * \brief
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
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MX_KERNEL_TEM_PARAMS                           \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler, \
        AscendC::Std::enable_if_t<                          \
            AscendC::Std::is_same_v<KernelMmadWithScaleMx, typename BlockMmad::DispatchPolicy::ScheduleType>>

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal()
    {}
    __aicore__ inline ~GemmUniversal()
    {}

    static constexpr bool weightNz = BlockMmad::weightNz;
    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    static constexpr bool isAtomicAdd = BlockMmad::DispatchPolicy::isAtomicAdd;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t kCacheLineAlignMask = IsFp4<AType>() ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = 2;

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;

    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::AuxGetC0Size<CType>()>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

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
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
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
    __aicore__ inline void ProcessSingleBatch(
        const Params& params, BlockScheduler& bs, uint64_t batchCnt, bool isTailRound);

    __aicore__ inline void ProcessWithBatch(const Params& params, BlockScheduler& bs);
    __aicore__ inline void AddBatchOffset(const Params& params);

    template <typename TensorB, typename TensorScaleB, typename TensorC>
    __aicore__ inline void SetL2Cache(
        const ProblemShape& problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB& gmB,
        TensorScaleB& gmScaleB, TensorC& gmC);

    template <typename TensorScaleB>
    __aicore__ inline void SetScaleL2Cache(
        const ProblemShape& problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB& gmScaleB);

private:
    BlockMmad mmadOp_;

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // 可选输入，直接初始化
    __gm__ fp8_e8m0_t* scaleAGmAddr_;
    __gm__ fp8_e8m0_t* scaleBGmAddr_;

    uint64_t blockIdx_;
    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    bool isBiasThreeDim_{false};
    bool isBias_{false};
    bool isSameBatch_{false};
    bool needUpdateTail_{false};
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::Run(const Params& params)
{
    if constexpr (isAtomicAdd) {
        AscendC::SetAtomicAdd<float>();
    }
    Init(params);
    BlockScheduler bs(params.problemShape, params.schParams);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
    mmadOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias_, enableL0CPingPong);

    if (AscendC::Te::Get<MNK_B>(params.problemShape) == 1) {
        ProcessSingleBatch(params, bs, 0, true);
        if constexpr (isAtomicAdd) {
            AscendC::SetAtomicNone();
        }
        return;
    }

    ProcessWithBatch(params, bs);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorScaleB>
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::SetScaleL2Cache(
    const ProblemShape& problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB& gmScaleB)
{
    if (AscendC::Te::Get<MNK_B>(problemShape) != 1) {
        return;
    }
    if constexpr (transB) {
        const int64_t scaleKRowBytes =
            Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_K>(problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        const int64_t scaleKL1RowBytes = Blaze::Gemm::CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // 0x7f: 128B cache line alignment for mx scale GM streaming
        const bool scaleAlignForL2Stream = (scaleKRowBytes & 0x7f) == 0 && (scaleKL1RowBytes & 0x7f) == 0;
        gmScaleB.SetL2CacheHint(
            scaleAlignForL2Stream ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                    AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
        const int64_t scaleNStrideBytes = AscendC::Te::Get<MNK_N>(problemShape) * MXFP_MULTI_BASE_SIZE;
        const int64_t scaleBaseNStrideBytes = baseN * MXFP_MULTI_BASE_SIZE;
        // 0x7f: 128B cache line alignment for mx scale GM streaming
        const bool scaleAlignForL2Stream = (scaleNStrideBytes & 0x7f) == 0 && (scaleBaseNStrideBytes & 0x7f) == 0;
        gmScaleB.SetL2CacheHint(
            scaleAlignForL2Stream ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                    AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorScaleB, typename TensorC>
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::SetL2Cache(
    const ProblemShape& problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB& gmB,
    TensorScaleB& gmScaleB, TensorC& gmC)
{
    if constexpr (isAtomicAdd) {
        gmC.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
    }

    const bool fullMTile = curBaseM >= AscendC::Te::Get<MNK_M>(problemShape);
    if (!(isSameBatch_ && fullMTile)) {
        return;
    }

    SetScaleL2Cache(problemShape, baseN, scaleKL1, gmScaleB);

    if constexpr (weightNz) {
        gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
    } else {
        // 0xff: 256 cache line alignment for FP4 weight GM streaming
        // 0x7f: 128 cache line alignment for FP8 weight GM streaming
        if constexpr (transB) {
            bool bAlignForL2Stream = (AscendC::Te::Get<MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(
                bAlignForL2Stream ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                    AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        } else {
            bool bAlignForL2Stream = (AscendC::Te::Get<MNK_N>(problemShape) & kCacheLineAlignMask) == 0 &&
                                     (baseN & kCacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(
                bAlignForL2Stream ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                    AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        }
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (params.qbmmParams.isBias == 1) {
        if (params.qbmmParams.biasThreeDim == 1) {
            isBiasThreeDim_ = true;
        }
        isBias_ = true;
    }
    if (params.qbmmParams.batchA1 == params.qbmmParams.batchB1 &&
        params.qbmmParams.batchA2 == params.qbmmParams.batchB2 &&
        params.qbmmParams.batchA3 == params.qbmmParams.batchB3 &&
        params.qbmmParams.batchA4 == params.qbmmParams.batchB4) {
        isSameBatch_ = true;
    }
    ResetGmAddr(params);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }

    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
    if (isBias_) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessWithBatch(
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
    uint64_t singleBatchTileCnt = bs.GetTotalCnt();
    uint64_t tailRoundStart =
        (singleBatchTileCnt * AscendC::Te::Get<MNK_B>(params.problemShape) / AscendC::GetBlockNum()) *
        AscendC::GetBlockNum();
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
                    bool isTailRound = curBatchC * singleBatchTileCnt > tailRoundStart;
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

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::AddBatchOffset(const Params& params)
{
    ResetGmAddr(params);
    constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
    aGmAddr_ +=
        (batchAOffset_ * AscendC::Te::Get<MNK_M>(params.problemShape) * AscendC::Te::Get<MNK_K>(params.problemShape)) >>
        sizeShift;
    if constexpr (weightNz) {
        if constexpr (transB) {
            bGmAddr_ +=
                (batchBOffset_ * Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_K>(params.problemShape), C0_SIZE) *
                 Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_N>(params.problemShape), static_cast<int64_t>(BLOCK_CUBE)) *
                 BLOCK_CUBE * C0_SIZE) >>
                sizeShift;
        } else {
            bGmAddr_ +=
                (batchBOffset_ * Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_N>(params.problemShape), C0_SIZE) *
                 Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(BLOCK_CUBE)) *
                 BLOCK_CUBE * C0_SIZE) >>
                sizeShift;
        }
    } else {
        bGmAddr_ += (batchBOffset_ * AscendC::Te::Get<MNK_N>(params.problemShape) *
                     AscendC::Te::Get<MNK_K>(params.problemShape)) >>
                    sizeShift;
    }
    cGmAddr_ +=
        batchCOffset_ * AscendC::Te::Get<MNK_M>(params.problemShape) * AscendC::Te::Get<MNK_N>(params.problemShape);
    if (isBiasThreeDim_) {
        biasGmAddr_ += batchCOffset_ * AscendC::Te::Get<MNK_N>(params.problemShape);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessSingleBatch(
    const Params& params, BlockScheduler& bs, uint64_t restBatch, bool isTailRound)
{
    auto scaleKLen =
        Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    auto layoutA =
        MakeLayoutA{}(AscendC::Te::Get<MNK_M>(params.problemShape), AscendC::Te::Get<MNK_K>(params.problemShape));
    auto layoutScaleA = MakeLayoutScaleA{}(AscendC::Te::Get<MNK_M>(params.problemShape), scaleKLen);
    auto layoutB =
        MakeLayoutB{}(AscendC::Te::Get<MNK_K>(params.problemShape), AscendC::Te::Get<MNK_N>(params.problemShape));
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, AscendC::Te::Get<MNK_N>(params.problemShape));
    auto layoutBias =
        AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1L, AscendC::Te::Get<MNK_N>(params.problemShape));
    auto layoutC =
        MakeLayoutC{}(AscendC::Te::Get<MNK_M>(params.problemShape), AscendC::Te::Get<MNK_N>(params.problemShape));

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
    auto gmScaleA =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleAGmAddr_), layoutScaleA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
    auto gmScaleB =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_), layoutBias);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);

    BlockCoord blockIdx;
    auto& mTailTile = params.schParams.mTailTile;
    auto& nTailTile = params.schParams.nTailTile;
    // both tail of current batch and rest batch are tail round
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                            AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    SetL2Cache(
        params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, params.l1Params.scaleKL1, gmB, gmScaleB,
        gmC);

    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L; // 不切K，所以坐标是0
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape =
            bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
        if (AscendC::Te::Get<IDX_M_TILEIDX>(singleShape) <= 0 || AscendC::Te::Get<IDX_N_TILEIDX>(singleShape) <= 0) {
            return;
        }

        bs.GetTileCoord(blockIdx, mPos, nPos);
        auto gmBlockA = gmA.Slice(
            AscendC::Te::MakeCoord(mPos, kPos),
            AscendC::Te::MakeShape(
                AscendC::Te::Get<IDX_M_TILEIDX>(singleShape), AscendC::Te::Get<MNK_K>(params.problemShape)));
        auto gmBlockScaleA = gmScaleA.Slice(
            AscendC::Te::MakeCoord(mPos, kPos),
            AscendC::Te::MakeShape(AscendC::Te::Get<IDX_M_TILEIDX>(singleShape), scaleKLen));
        auto gmBlockB = gmB.Slice(
            AscendC::Te::MakeCoord(kPos, nPos),
            AscendC::Te::MakeShape(
                AscendC::Te::Get<MNK_K>(params.problemShape), AscendC::Te::Get<IDX_N_TILEIDX>(singleShape)));
        auto gmBlockScaleB = gmScaleB.Slice(
            AscendC::Te::MakeCoord(kPos, nPos),
            AscendC::Te::MakeShape(scaleKLen, AscendC::Te::Get<IDX_N_TILEIDX>(singleShape)));
        auto gmBlockBias = gmBias.Slice(
            AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(1L, AscendC::Te::Get<IDX_N_TILEIDX>(singleShape)));
        auto gmBlockC = gmC.Slice(
            AscendC::Te::MakeCoord(mPos, nPos),
            AscendC::Te::MakeShape(
                AscendC::Te::Get<IDX_M_TILEIDX>(singleShape), AscendC::Te::Get<IDX_N_TILEIDX>(singleShape)));
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
    }
    bs.UpdateNextBatchBlockRoundParams();
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
