/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef NPU_MEGA_MOE_HCCL_COMMON_H
#define NPU_MEGA_MOE_HCCL_COMMON_H

#include "aclnn_torch_adapter/op_api_common.h"
#include <acl/acl_rt.h>
#include <dlfcn.h>
#include <vector>
#include <c10/util/Exception.h>
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hccl/hcomm_res_defs.h"

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

// Function pointer types for CHANNEL mode (from libhccl_fwk.so)

// Get comm handle by group name
using _HcomGetCommHandleByGroup = HcclResult (*)(const char *, HcclComm *);

// Get links between ranks
using _HcclRankGraphGetLinks = HcclResult (*)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink **, uint32_t *);

// Get network layers
using _HcclRankGraphGetLayers = HcclResult (*)(HcclComm, uint32_t **, uint32_t *);

// Get rank size per layer
using _HcclRankGraphGetRankSizeByLayer = HcclResult (*)(HcclComm, uint32_t, uint32_t *);

// Get ranks by layer
using _HcclRankGraphGetRanksByLayer = HcclResult (*)(HcclComm, uint32_t, uint32_t **, uint32_t *);

// Acquire HCCL channel
using _HcclChannelAcquire = HcclResult (*)(HcclComm, CommEngine, HcclChannelDesc *, uint32_t, ChannelHandle *);

// Get HCCL buffer via channel
using _HcclChannelGetHcclBuffer = HcclResult (*)(HcclComm, ChannelHandle, void **, uint64_t *);

// Create engine context
using _HcclEngineCtxCreate = HcclResult (*)(HcclComm, const char *, CommEngine, uint64_t, void **);

// Get engine context
using _HcclEngineCtxGet = HcclResult (*)(HcclComm, const char *, CommEngine, void **, uint64_t *);

// Copy context
using _HcclEngineCtxCopy = HcclResult (*)(HcclComm, CommEngine, const char *, void *, uint64_t, uint64_t);

// Get HCCL buffer (local)
using _HcclGetHcclBuffer = HcclResult (*)(HcclComm, void **, uint64_t *);

// Get rank ID
using _HcclGetRankId = HcclResult (*)(HcclComm, uint32_t *);

// Get rank size
using _HcclGetRankSize = HcclResult (*)(HcclComm, uint32_t *);

// Global function pointers (initialized by InitHcclEngineCtxFunctions)
static _HcomGetCommHandleByGroup HcomGetCommHandleByGroupFunc = nullptr;
static _HcclRankGraphGetLinks HcclRankGraphGetLinksFunc = nullptr;
static _HcclRankGraphGetLayers HcclRankGraphGetLayersFunc = nullptr;
static _HcclRankGraphGetRankSizeByLayer HcclRankGraphGetRankSizeByLayerFunc = nullptr;
static _HcclRankGraphGetRanksByLayer HcclRankGraphGetRanksByLayerFunc = nullptr;
static _HcclChannelAcquire HcclChannelAcquireFunc = nullptr;
static _HcclChannelGetHcclBuffer HcclChannelGetHcclBufferFunc = nullptr;
static _HcclEngineCtxCreate HcclEngineCtxCreateFunc = nullptr;
static _HcclEngineCtxGet HcclEngineCtxGetFunc = nullptr;
static _HcclEngineCtxCopy HcclEngineCtxCopyFunc = nullptr;
static _HcclGetHcclBuffer HcclGetHcclBufferFunc = nullptr;
static _HcclGetRankId HcclGetRankIdFunc = nullptr;
static _HcclGetRankSize HcclGetRankSizeFunc = nullptr;

template <typename T>
inline T GetHcclFwkFuncAddr(const char *apiName)
{
    static auto opApiHandler = GetOpApiLibHandler("libhccl_fwk.so");
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, "libhccl_fwk.so", apiName);
    if (funcAddr == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<T>(funcAddr);
}

inline void InitHcclEngineCtxFunctions()
{
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    HcomGetCommHandleByGroupFunc = GetHcclFwkFuncAddr<_HcomGetCommHandleByGroup>("HcomGetCommHandleByGroup");
    TORCH_CHECK(HcomGetCommHandleByGroupFunc != nullptr, "get HcomGetCommHandleByGroup failed.");

    HcclRankGraphGetLinksFunc = GetHcclFwkFuncAddr<_HcclRankGraphGetLinks>("HcclRankGraphGetLinks");
    TORCH_CHECK(HcclRankGraphGetLinksFunc != nullptr, "get HcclRankGraphGetLinks failed.");

    HcclRankGraphGetLayersFunc = GetHcclFwkFuncAddr<_HcclRankGraphGetLayers>("HcclRankGraphGetLayers");
    TORCH_CHECK(HcclRankGraphGetLayersFunc != nullptr, "get HcclRankGraphGetLayers failed.");

    HcclRankGraphGetRankSizeByLayerFunc =
        GetHcclFwkFuncAddr<_HcclRankGraphGetRankSizeByLayer>("HcclRankGraphGetRankSizeByLayer");
    TORCH_CHECK(HcclRankGraphGetRankSizeByLayerFunc != nullptr, "get HcclRankGraphGetRankSizeByLayer failed.");

    HcclRankGraphGetRanksByLayerFunc =
        GetHcclFwkFuncAddr<_HcclRankGraphGetRanksByLayer>("HcclRankGraphGetRanksByLayer");
    TORCH_CHECK(HcclRankGraphGetRanksByLayerFunc != nullptr, "get HcclRankGraphGetRanksByLayer failed.");

    HcclChannelAcquireFunc = GetHcclFwkFuncAddr<_HcclChannelAcquire>("HcclChannelAcquire");
    TORCH_CHECK(HcclChannelAcquireFunc != nullptr, "get HcclChannelAcquire failed.");

    HcclChannelGetHcclBufferFunc = GetHcclFwkFuncAddr<_HcclChannelGetHcclBuffer>("HcclChannelGetHcclBuffer");
    TORCH_CHECK(HcclChannelGetHcclBufferFunc != nullptr, "get HcclChannelGetHcclBuffer failed.");

    HcclEngineCtxCreateFunc = GetHcclFwkFuncAddr<_HcclEngineCtxCreate>("HcclEngineCtxCreate");
    TORCH_CHECK(HcclEngineCtxCreateFunc != nullptr, "get HcclEngineCtxCreate failed.");

    HcclEngineCtxGetFunc = GetHcclFwkFuncAddr<_HcclEngineCtxGet>("HcclEngineCtxGet");
    TORCH_CHECK(HcclEngineCtxGetFunc != nullptr, "get HcclEngineCtxGet failed.");

    HcclEngineCtxCopyFunc = GetHcclFwkFuncAddr<_HcclEngineCtxCopy>("HcclEngineCtxCopy");
    TORCH_CHECK(HcclEngineCtxCopyFunc != nullptr, "get HcclEngineCtxCopy failed.");

    HcclGetHcclBufferFunc = GetHcclFwkFuncAddr<_HcclGetHcclBuffer>("HcclGetHcclBuffer");
    TORCH_CHECK(HcclGetHcclBufferFunc != nullptr, "get HcclGetHcclBuffer failed.");

    HcclGetRankIdFunc = GetHcclFwkFuncAddr<_HcclGetRankId>("HcclGetRankId");
    TORCH_CHECK(HcclGetRankIdFunc != nullptr, "get HcclGetRankId failed.");

    HcclGetRankSizeFunc = GetHcclFwkFuncAddr<_HcclGetRankSize>("HcclGetRankSize");
    TORCH_CHECK(HcclGetRankSizeFunc != nullptr, "get HcclGetRankSize failed.");

}

#endif // NPU_MEGA_MOE_HCCL_COMMON_H
