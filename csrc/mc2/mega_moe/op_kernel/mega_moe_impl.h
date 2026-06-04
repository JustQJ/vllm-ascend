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
 * \file mega_moe_impl.h
 * \brief
 */

#ifndef MEGA_MOE_IMPL_H
#define MEGA_MOE_IMPL_H

#include "kernel_operator.h"

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_base.h"

#include "tensor_api/include/tensor_api/tensor.h"
#include "blaze/block/block_mmad_mx.h"
#include "blaze/block/block_scheduler_swizzle.h"

#include "mega_moe_combine_send.h"

using namespace AscendC;
using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;

namespace MegaMoeImpl {
using BlockScheduler = typename Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>;  // 3: SwizzleOffset
constexpr uint32_t ALIGN32 = 32U;
constexpr uint32_t L1_TILE_M = 256;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
constexpr uint32_t L0_TILE_K = 128;
constexpr uint32_t SCALE_K_L1_RATE = 2;
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr uint32_t MAX_SINGLE_MN = 256 * 256;
constexpr uint32_t BUFALIGN = (MAX_SINGLE_MN + 31U) / ALIGN32 * ALIGN32 * sizeof(bfloat16_t);
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM = (MAX_SINGLE_MN + 31U) / ALIGN32 * ALIGN32;

// GroupMatmulSwigluQuant - GMM1 + SwiGLU + Quant
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmulSwigluQuant(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>& epilogueOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom)
{
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<N_VALUE>(problemShape);
    uint32_t k = Get<K_VALUE>(problemShape);
    uint32_t outputN = n / SWIGLU_N_HALF;

    epilogueOp.UpdateNextProblem({m, outputN, k, 0});

    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx() / GetTaskRation();

    GlobalTensor<int32_t> groupFlagListGm2;
    groupFlagListGm2.SetGlobalBuffer((__gm__ int32_t*)gmmAddrInfo.groupFlagList2);

    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

    constexpr bool transA = false;
    constexpr bool transB = true; // 后续要改为从模板参数获取转置属性

    constexpr uint32_t c0SizeA = AuxGetC0Size<ElementA>();
    constexpr uint32_t c0SizeB = AuxGetC0Size<ElementB>();
    constexpr uint32_t c0SizeC = AuxGetC0Size<ElementC>();
    constexpr uint32_t c0SizeScale = 2U;

    using LayoutA = Std::conditional_t<transA, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<transB, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<c0SizeA>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<c0SizeB>>;
    using MakeLayoutScaleA = Std::conditional_t<
        transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeC>>;

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleK);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleK, n);
    auto layoutBias = Te::MakeFrameLayout<LayoutBias>(1u, n); // block mmad需要传入bias
    auto layoutC = MakeLayoutC{}(L1_TILE_M, L1_TILE_N);

    using BiasType = float;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, BiasType, LayoutBias, void>;
    BlockMmad blockMmad;
    bool enableL0CPingPong = false;
    typename BlockMmad::L1Params l1Params{
        .kL1 = L1_TILE_K,
        .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE,
        .l1BufNum = 2 // 2: double buffer
    };
    typename BlockMmad::BlockShape l0TileShape{L1_TILE_M, L1_TILE_N, L0_TILE_K, 0};
    typename BlockMmad::TupleShape matmulShape{m, n, k};
    blockMmad.Init(matmulShape, l0TileShape, l1Params, false, enableL0CPingPong); // 当前固定无bias

    int64_t ubOffset = 0;
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    ubOffset += MAX_SINGLE_MN_ALIGN32_NUM * sizeof(ElementC);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    GlobalTensor<ElementA> aGlobalTensor;
    GlobalTensor<ElementB> bGlobalTensor;

    aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal));
    bGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal));

    // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
    // data reads will bypass L2 Cache.
    if (CeilDiv(m, L1_TILE_M) == 1) {
        bGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    } else {
        bGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    }
    if (CeilDiv(n, L1_TILE_N) == 1) {
        aGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    } else {
        aGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    }

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layoutA);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layoutB);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layoutScaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layoutBias);

    BlockScheduler scheduler(
        {m, outputN, k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;

    // wave-grain dispatch flag: 每 wave (L1_TILE_M 行) 一个槽，dispatch 完成该 wave 内行数累加。
    // 目标值 = 该 wave 实际行数 = min(L1_TILE_M, m - mLoc)（尾 wave 可能 < L1_TILE_M）。
    // 仅在 wave 切换时等待，跨 nLoc 不同的同 mLoc tile 复用上次等待结果。
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        if constexpr (g_coreType == AscendC::AIC) {
            uint32_t waveIdx = mLoc / L1_TILE_M;
            if (waveIdx != lastWaveWaited) {
                uint32_t targetValue = (mLoc + L1_TILE_M > m) ? (m - mLoc) : L1_TILE_M;
                __gm__ int32_t* flagValueAddr =
                    (__gm__ int32_t*)groupFlagListGm2.GetPhyAddr() + waveIdx;
                while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {
                    }
                }
                lastWaveWaited = waveIdx;
            }

            if (vecSetSyncCom) {
                WaitForVector();
            }

            auto gmBlockA = gmA.Slice(
                Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
            auto gmBlockScaleA = gmScaleA.Slice(
                Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

            auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));

            typename BlockMmad::BlockShape singleShape{
                Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto gmBlockB = gmB.Slice(
                    Te::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                    Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                auto gmBlockScaleB = gmScaleB.Slice(
                    Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
                    Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
                blockMmad(
                    gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                    weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
            }

            NotifyVector();
        }

        vecSetSyncCom = 1;

        if constexpr (g_coreType == AscendC::AIV) {
            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{
                Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                mLoc * outputN + nLoc,
                mLoc * CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                0, 0, 0, 0};
            WaitForCube();
            AscendC::SetCtrlSpr<60, 60>(0);
            epilogueOp(epilogueShape, epilogueOffset);
            NotifyCube();
        }
    }
    startBlockIdx = (startBlockIdx + tileNum) % blockNum;
}

// GroupMatmul2 - GMM2矩阵乘法 + Combine
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmul2(
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom2, uint32_t groupIdx,
    uint16_t& pingpongIdx, int32_t rankId)
{
    constexpr uint32_t GMM2_BLOCK_SIZE = L1_TILE_M * L1_TILE_N;

    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<K_VALUE>(problemShape);
    uint32_t k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;

    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx() / GetTaskRation();

    GlobalTensor<int32_t> groupFlagListGm;
    groupFlagListGm.SetGlobalBuffer((__gm__ int32_t*)gmmAddrInfo.groupFlagList);

    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

    constexpr bool transA = false;
    constexpr bool transB = true; // 后续要改为从模板参数获取转置属性

    constexpr uint32_t c0SizeA = AuxGetC0Size<ElementA>();
    constexpr uint32_t c0SizeB = AuxGetC0Size<ElementB>();
    constexpr uint32_t c0SizeC = AuxGetC0Size<ElementC>();
    constexpr uint32_t c0SizeScale = 2U;

    using LayoutA = Std::conditional_t<transA, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutB = Std::conditional_t<transB, Te::DNExtLayoutPtn, Te::NDExtLayoutPtn>;
    using LayoutBias = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<c0SizeA>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<c0SizeB>>;
    using MakeLayoutScaleA = Std::conditional_t<
        transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutScaleB = Std::conditional_t<
        transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<c0SizeScale>>,
        Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<c0SizeScale>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<c0SizeC>>;

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleK);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleK, n);
    auto layoutBias = Te::MakeFrameLayout<LayoutBias>(1U, n); // block mmad需要传入bias
    auto layoutC = MakeLayoutC{}(L1_TILE_M, L1_TILE_N);

    using BiasType = float;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, BiasType, LayoutBias, void>;
    BlockMmad blockMmad;
    bool enableL0CPingPong = false;
    typename BlockMmad::L1Params l1Params{
        .kL1 = L1_TILE_K,
        .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE,
        .l1BufNum = 2 // 2: double buffer
    };
    typename BlockMmad::BlockShape l0TileShape{L1_TILE_M, L1_TILE_N, L0_TILE_K, 0};
    typename BlockMmad::TupleShape matmulShape{m, n, k};
    blockMmad.Init(matmulShape, l0TileShape, l1Params, false, enableL0CPingPong); // 当前固定无bias

    int64_t ubOffset = 0;
    LocalTensor<ElementC> l0cOutUbGMM2First = LocalTensor<ElementC>(TPosition::VECIN, ubOffset, L1_TILE_M * L1_TILE_N);
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    ubOffset += MAX_SINGLE_MN_ALIGN32_NUM * sizeof(ElementC);
    LocalTensor<ElementC> l0cOutUbGMM2Second = LocalTensor<ElementC>(TPosition::VECIN, ubOffset, L1_TILE_M * L1_TILE_N);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutC);

    GlobalTensor<ElementA> aGlobalTensor;
    GlobalTensor<ElementB> bGlobalTensor;

    aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal));
    bGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal));

    // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
    // data reads will bypass L2 Cache.
    if (CeilDiv(m, L1_TILE_M) == 1) {
        bGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    } else {
        bGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    }
    if (CeilDiv(n, L1_TILE_N) == 1) {
        aGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    } else {
        aGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    }

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layoutA);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layoutB);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layoutScaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layoutBias);

    BlockScheduler scheduler(
        {m, n, k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M), static_cast<int64_t>(L1_TILE_N))});
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;
    uint32_t tileNum = scheduler.GetTileNum();

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        auto tensorUb = pingpongIdx == 0 ? l0cOutUbFirst : l0cOutUbSecond;
        auto l0cOutUbGMM2 = pingpongIdx == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;

        if constexpr (g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                BlockScheduler gmmBlockScheduler(
                    {m, k, n},
                    BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M),
                        static_cast<int64_t>(L1_TILE_N))});
                uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
                __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm.GetPhyAddr();
                while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {
                    }
                }
            }
            if (vecSetSyncCom2 >= 2) {
                WaitForVector(pingpongIdx);
            }

            auto gmBlockA = gmA.Slice(
                Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
            auto gmBlockScaleA = gmScaleA.Slice(
                Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

            auto tensorBlockUb = tensorUb.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));

            auto gmBlockB = gmB.Slice(
                Te::MakeCoord(kLoc, nLoc), Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = gmScaleB.Slice(
                Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));

            typename BlockMmad::BlockShape singleShape{
                Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockUb, singleShape);

            NotifyVector(pingpongIdx);
        }

        vecSetSyncCom2++;

        if constexpr (g_coreType == AscendC::AIV) {
            WaitForCube(pingpongIdx);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            MegaMoeCombineImpl::CombineTokens<ElementC, decltype(actualShape)>(
                mLoc, nLoc, n, groupIdx, rankId, l0cOutUbGMM2, actualShape, params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            NotifyCube(pingpongIdx);
        }
        pingpongIdx = 1 - pingpongIdx;
    }
    startBlockIdx = (startBlockIdx + tileNum) % blockNum;
}

} // namespace MegaMoeImpl

#endif