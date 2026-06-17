/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MEGA_MOE_HCCL_COMMON_H
#define MEGA_MOE_HCCL_COMMON_H

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "hccl/hccl_res.h"
#include "hccl/hcomm_res_defs.h"
#include "hccl/hccl_rank_graph.h"

namespace vllm_ascend {

// ======================== Log macro ========================
#define ASCEND_LOGW(fmt, ...) \
    fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__)
#define ASCEND_LOGI(fmt, ...) \
    fprintf(stderr, "[INFO] " fmt "\n", ##__VA_ARGS__)

// ======================== HCCL Function pointer types ========================

// KFC mode function pointers
using _HcclKfcAllocOpArgs = HcclResult (*)(void **);
using _HcclKfcOpArgsSetAlgConfig = HcclResult (*)(void *, char *);
using _HcclKfcOpArgsSetCommEngine = HcclResult (*)(void *, uint8_t);
using _HcclCreateOpResCtx = HcclResult (*)(HcclComm, uint8_t, void *, void **);
using _HcclGetRemoteIpcHcclBuf = HcclResult (*)(HcclComm, uint64_t, void **, uint64_t *);
using _HcclKfcFreeOpArgs = HcclResult (*)(void *);
using _HcclCommGetHandleWithName = HcclResult (*)(const char *, HcclComm*);
using _HcclGetRankSize = HcclResult (*)(HcclComm, uint32_t *);
using _HcclGetRankId = HcclResult (*)(HcclComm, uint32_t *);
using _HcclGetHcclBuffer = HcclResult (*)(HcclComm, void **, uint64_t *);

// HCCL EngineCtx (Channel mode) function pointers
using _HcomGetCommHandleByGroup = HcclResult (*)(const char*, HcclComm*);
using _HcclRankGraphGetLinks = HcclResult (*)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*);
using _HcclRankGraphGetLayers = HcclResult (*)(HcclComm, uint32_t**, uint32_t*);
using _HcclRankGraphGetRankSizeByLayer = HcclResult (*)(HcclComm, uint32_t, uint32_t*);
using _HcclRankGraphGetRanksByLayer = HcclResult (*)(HcclComm, uint32_t, uint32_t **, uint32_t *);
using _HcclChannelDescInit = void (*)(HcclChannelDesc*, uint32_t);
using _HcclChannelAcquire = HcclResult (*)(HcclComm, CommEngine, HcclChannelDesc*, uint32_t, ChannelHandle*);
using _HcclChannelGetHcclBuffer = HcclResult (*)(HcclComm, ChannelHandle, void**, uint64_t*);
using _HcclEngineCtxCreate = HcclResult (*)(HcclComm, const char*, CommEngine, uint64_t, void**);
using _HcclEngineCtxGet = HcclResult (*)(HcclComm, const char*, CommEngine, void**, uint64_t*);
using _HcclEngineCtxCopy = HcclResult (*)(HcclComm, CommEngine, const char*, void*, uint64_t, uint64_t);

// ======================== Static function pointers ========================

// KFC mode
static _HcclKfcAllocOpArgs HcclKfcAllocOpArgsFunc = nullptr;
static _HcclKfcOpArgsSetAlgConfig HcclKfcOpArgsSetAlgConfigFunc = nullptr;
static _HcclKfcOpArgsSetCommEngine HcclKfcOpArgsSetCommEngineFunc = nullptr;
static _HcclCreateOpResCtx HcclCreateOpResCtxFunc = nullptr;
static _HcclGetRemoteIpcHcclBuf HcclGetRemoteIpcHcclBufFunc = nullptr;
static _HcclKfcFreeOpArgs HcclKfcFreeOpArgsFunc = nullptr;
static _HcclCommGetHandleWithName HcclCommGetHandleWithNameFunc = nullptr;
static _HcclGetRankSize HcclGetRankSizeFunc = nullptr;
static _HcclGetRankId HcclGetRankIdFunc = nullptr;
static _HcclGetHcclBuffer HcclGetHcclBufferFunc = nullptr;

// Channel mode
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

// ======================== Library helpers ========================

inline const char *GetHcclLibName(void) { return "libhccl.so"; }
inline const char *GetHcclFwkLibName(void) { return "libhccl_fwk.so"; }

inline void *GetOpApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s", libName, dlerror());
    }
    return handler;
}

template <typename T>
inline T GetFuncAddr(void *opApiHandler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(opApiHandler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s", apiName, libName, dlerror());
        return nullptr;
    }
    return reinterpret_cast<T>(funcAddr);
}

template <typename T>
inline T GetHcclFuncAddr(const char *apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetHcclLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetFuncAddr<T>(opApiHandler, GetHcclLibName(), apiName);
}

template <typename T>
inline T GetHcclFwkFuncAddr(const char *apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetHcclFwkLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetFuncAddr<T>(opApiHandler, GetHcclFwkLibName(), apiName);
}

// ======================== Initialize HCCL function pointers ========================

inline void InitHcclFunctions()
{
    HcclKfcAllocOpArgsFunc = GetHcclFuncAddr<_HcclKfcAllocOpArgs>("HcclKfcAllocOpArgs");
    TORCH_CHECK(HcclKfcAllocOpArgsFunc != nullptr, "getHcclKfcAllocOpArgs failed.");
    HcclKfcFreeOpArgsFunc = GetHcclFuncAddr<_HcclKfcFreeOpArgs>("HcclKfcFreeOpArgs");
    TORCH_CHECK(HcclKfcFreeOpArgsFunc != nullptr, "getHcclKfcFreeOpArgs failed.");
    HcclKfcOpArgsSetCommEngineFunc =
        GetHcclFuncAddr<_HcclKfcOpArgsSetCommEngine>("HcclKfcOpArgsSetCommEngine");
    TORCH_CHECK(HcclKfcOpArgsSetCommEngineFunc != nullptr, "getHcclKfcOpArgsSetCommEngine failed.");
    HcclGetRankIdFunc = GetHcclFuncAddr<_HcclGetRankId>("HcclGetRankId");
    TORCH_CHECK(HcclGetRankIdFunc != nullptr, "getFuncHcclGetRankId failed.");
    HcclGetHcclBufferFunc = GetHcclFuncAddr<_HcclGetHcclBuffer>("HcclGetHcclBuffer");
    TORCH_CHECK(HcclGetHcclBufferFunc != nullptr, "getFuncHcclGetHcclBuffer failed.");
    HcclGetRemoteIpcHcclBufFunc = GetHcclFwkFuncAddr<_HcclGetRemoteIpcHcclBuf>("HcclGetRemoteIpcHcclBuf");
    TORCH_CHECK(HcclGetRemoteIpcHcclBufFunc != nullptr, "getFuncHcclGetRemoteIpcHcclBuf failed.");
    HcclKfcOpArgsSetAlgConfigFunc = GetHcclFuncAddr<_HcclKfcOpArgsSetAlgConfig>("HcclKfcOpArgsSetAlgConfig");
    TORCH_CHECK(HcclKfcOpArgsSetAlgConfigFunc != nullptr, "getFuncHcclKfcOpArgsSetAlgConfig failed.");
    HcclCommGetHandleWithNameFunc =
        GetHcclFwkFuncAddr<_HcclCommGetHandleWithName>("HcclCommGetHandleWithName");
    TORCH_CHECK(HcclCommGetHandleWithNameFunc != nullptr, "getFuncHcclCommGetHandleWithName failed.");
    HcclCreateOpResCtxFunc = GetHcclFuncAddr<_HcclCreateOpResCtx>("HcclCreateOpResCtx");
    TORCH_CHECK(HcclCreateOpResCtxFunc != nullptr, "getFuncHcclCreateOpResCtx failed.");
    HcclGetRankSizeFunc = GetHcclFuncAddr<_HcclGetRankSize>("HcclGetRankSize");
    TORCH_CHECK(HcclGetRankSizeFunc != nullptr, "getFuncHcclGetRankSize failed.");
}

inline void InitHcclEngineCtxFunctions()
{
    HcomGetCommHandleByGroupFunc = GetHcclFwkFuncAddr<_HcomGetCommHandleByGroup>("HcomGetCommHandleByGroup");
    TORCH_CHECK(HcomGetCommHandleByGroupFunc != nullptr, "getHcomGetCommHandleByGroup failed.");
    HcclRankGraphGetLinksFunc = GetHcclFwkFuncAddr<_HcclRankGraphGetLinks>("HcclRankGraphGetLinks");
    TORCH_CHECK(HcclRankGraphGetLinksFunc != nullptr, "getHcclRankGraphGetLinks failed.");
    HcclRankGraphGetLayersFunc = GetHcclFwkFuncAddr<_HcclRankGraphGetLayers>("HcclRankGraphGetLayers");
    TORCH_CHECK(HcclRankGraphGetLayersFunc != nullptr, "getHcclRankGraphGetLayers failed.");
    HcclRankGraphGetRankSizeByLayerFunc = GetHcclFwkFuncAddr<_HcclRankGraphGetRankSizeByLayer>(
        "HcclRankGraphGetRankSizeByLayer");
    TORCH_CHECK(HcclRankGraphGetRankSizeByLayerFunc != nullptr, "getHcclRankGraphGetRankSizeByLayer failed.");
    HcclRankGraphGetRanksByLayerFunc = GetHcclFwkFuncAddr<_HcclRankGraphGetRanksByLayer>(
        "HcclRankGraphGetRanksByLayer");
    TORCH_CHECK(HcclRankGraphGetRanksByLayerFunc != nullptr, "getHcclRankGraphGetRanksByLayer failed.");
    HcclChannelAcquireFunc = GetHcclFwkFuncAddr<_HcclChannelAcquire>("HcclChannelAcquire");
    TORCH_CHECK(HcclChannelAcquireFunc != nullptr, "getHcclChannelAcquire failed.");
    HcclChannelGetHcclBufferFunc = GetHcclFwkFuncAddr<_HcclChannelGetHcclBuffer>("HcclChannelGetHcclBuffer");
    TORCH_CHECK(HcclChannelGetHcclBufferFunc != nullptr, "getHcclChannelGetHcclBuffer failed.");
    HcclEngineCtxCreateFunc = GetHcclFwkFuncAddr<_HcclEngineCtxCreate>("HcclEngineCtxCreate");
    TORCH_CHECK(HcclEngineCtxCreateFunc != nullptr, "getHcclEngineCtxCreate failed.");
    HcclEngineCtxGetFunc = GetHcclFwkFuncAddr<_HcclEngineCtxGet>("HcclEngineCtxGet");
    TORCH_CHECK(HcclEngineCtxGetFunc != nullptr, "getHcclEngineCtxGet failed.");
    HcclEngineCtxCopyFunc = GetHcclFwkFuncAddr<_HcclEngineCtxCopy>("HcclEngineCtxCopy");
    TORCH_CHECK(HcclEngineCtxCopyFunc != nullptr, "getHcclEngineCtxCopy failed.");
    HcclGetHcclBufferFunc = GetHcclFwkFuncAddr<_HcclGetHcclBuffer>("HcclGetHcclBuffer");
    TORCH_CHECK(HcclGetHcclBufferFunc != nullptr, "getFuncHcclGetHcclBuffer failed.");
    HcclGetRankIdFunc = GetHcclFwkFuncAddr<_HcclGetRankId>("HcclGetRankId");
    TORCH_CHECK(HcclGetRankIdFunc != nullptr, "getFuncHcclGetRankId failed.");
    HcclGetRankSizeFunc = GetHcclFwkFuncAddr<_HcclGetRankSize>("HcclGetRankSize");
    TORCH_CHECK(HcclGetRankSizeFunc != nullptr, "getFuncHcclGetRankSize failed.");
}

} // namespace vllm_ascend

#endif // MEGA_MOE_HCCL_COMMON_H
