/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 *
 * mega_moe_torch_adpt.h - Torch adapter for MegaMoe operator.
 * Internally constructs the HCCL peermem context from the EP group name,
 * then delegates to the CANN ACLNN kernel.
 */

#ifndef MEGA_MOE_TORCH_ADPT_H
#define MEGA_MOE_TORCH_ADPT_H

#include <torch/extension.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "hccl_common.h"
#include "op_host/op_api/aclnn_mega_moe.h"
#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace vllm_ascend {

constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint64_t CONTEXT_COPY_OFFSET = 0;
constexpr int64_t DISPATCH_QUANT_MODE_MXFP = 4;
constexpr int64_t TORCH_DTYPE_FLOAT8_E5M2 = 23;
constexpr int64_t TORCH_DTYPE_FLOAT8_E4M3FN = 24;

// ---- Mc2MoeContext struct (layout must match kernel-side Mc2MoeContext in mega_moe_base.h) ----
struct Mc2MoeContext {
    uint32_t epRankId = 0;
    uint32_t epRankSize = 0;
    uint64_t winSize = 0;
    uint64_t epHcclBuffer[HCCL_MAX_RANK_SIZE] = {};
};

struct MegaMoeRuntimeContext {
    at::Tensor context;
    int64_t epWorldSize = 0;
    int64_t cclBufferSize = 0;
};

// ---- HCCL Channel Context Builder ----
class MegaMoeContextBuilder {
public:
    MegaMoeRuntimeContext Build(c10::string_view group_ep) {
        CheckAscend950Only();
        InitHcclEngineCtxFunctions();

        const std::string groupName(group_ep);
        HcclComm hcclHandle = nullptr;
        HcclResult ret = HcomGetCommHandleByGroupFunc(groupName.c_str(), &hcclHandle);
        TORCH_CHECK(ret == HCCL_SUCCESS, "Get HCCL handle failed for group: ", groupName);

        CommProtocol protocol;
        GetCommProtocol(hcclHandle, protocol);

        Mc2MoeContext ctx{};
        uint64_t hcclBuffSize = 0;
        CreateContext(hcclHandle, groupName, protocol, ctx, hcclBuffSize);

        at::Tensor context = at::empty(
            {static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t))},
            at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1));
        at::Tensor hostCtx = at::from_blob(
            &ctx, {static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t))}, at::kInt);
        context.copy_(hostCtx);
        return {context, static_cast<int64_t>(ctx.epRankSize), static_cast<int64_t>(hcclBuffSize)};
    }

private:
    static void CheckAscend950Only() {
        const char *socName = aclrtGetSocName();
        TORCH_CHECK(socName != nullptr && std::strstr(socName, "Ascend950") != nullptr,
                    "npu_mega_moe only supports Ascend950 channel mode, got SoC: ",
                    socName == nullptr ? "nullptr" : socName);
    }

    void GetCommProtocol(const HcclComm &commHandle, CommProtocol &protocol) {
        uint32_t layerNum = 0;
        uint32_t *layerList = nullptr;
        auto ret = HcclRankGraphGetLayersFunc(commHandle, &layerList, &layerNum);
        TORCH_CHECK(ret == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", ret);

        if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
            protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
            return;
        }

        CheckProtocolSupport(commHandle, layerList, layerNum);
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    }

    void CheckProtocolSupport(const HcclComm &commHandle, uint32_t *&layerList, uint32_t &layerNum) {
        uint32_t srcRankId = 0;
        auto hcclRet = HcclGetRankIdFunc(commHandle, &srcRankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "CheckProtocolSupport: Get rank ID failed, ret: ", hcclRet);

        for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
            uint32_t *rankIdLists = nullptr;
            uint32_t rankNumInLayer = 0;
            hcclRet = HcclRankGraphGetRanksByLayerFunc(
                commHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank IDs by layer failed, ret: ", hcclRet);

            for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
                if (rankIdLists[rankId] == srcRankId ||
                    layerMap_.find(rankIdLists[rankId]) != layerMap_.end()) {
                    continue;
                }

                uint32_t netLinkNum = 0;
                CommLink *linksList = nullptr;
                hcclRet = HcclRankGraphGetLinksFunc(
                    commHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                    &linksList, &netLinkNum);
                TORCH_CHECK(hcclRet == HCCL_SUCCESS,
                            "Get HCCL links failed when checking protocol support, ret: ", hcclRet);
                TORCH_CHECK(CheckLinks(netLinkNum, linksList),
                            "No HCCL links support UB_MEM, srcRank ", srcRankId,
                            ", dstRank ", rankIdLists[rankId], ", layer ", layerList[layerIndex]);

                layerMap_[rankIdLists[rankId]] = layerList[layerIndex];
            }
        }
    }

    static bool CheckLinks(uint32_t netLinkNum, CommLink *linksList) {
        for (uint32_t j = 0; j < netLinkNum; ++j) {
            if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
                return true;
            }
        }
        return false;
    }

    void CreateContext(const HcclComm &commHandle, const std::string &tag,
                       const CommProtocol &protocol, Mc2MoeContext &ctx, uint64_t &hcclBuffSize) {
        std::string mc2ContextTag = tag + "mega_moe";
        TORCH_CHECK(mc2ContextTag.size() <= 255, "Context tag too long");

        CommEngine engine = CommEngine::COMM_ENGINE_AIV;
        void *deviceCtx = nullptr;

        auto hcclRet = HcclGetRankIdFunc(commHandle, &ctx.epRankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);

        uint32_t rankSize = 0;
        hcclRet = HcclGetRankSizeFunc(commHandle, &rankSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);
        TORCH_CHECK(rankSize > 1 && rankSize <= HCCL_MAX_RANK_SIZE,
                    "Invalid EP rank size for npu_mega_moe: ", rankSize);
        ctx.epRankSize = rankSize;

        GetHcclCommResource(commHandle, engine, protocol, ctx, rankSize, hcclBuffSize);

        uint64_t ctxSize = 0;
        hcclRet = HcclEngineCtxGetFunc(commHandle, mc2ContextTag.c_str(), engine, &deviceCtx, &ctxSize);
        if (hcclRet != HCCL_SUCCESS) {
            ctxSize = sizeof(Mc2MoeContext);
            hcclRet = HcclEngineCtxCreateFunc(commHandle, mc2ContextTag.c_str(),
                                              engine, ctxSize, &deviceCtx);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create HCCL context memory failed, ret: ", hcclRet);
        } else {
            TORCH_CHECK(ctxSize == sizeof(Mc2MoeContext),
                        "Cached mega_moe context size mismatch, expected ", sizeof(Mc2MoeContext),
                        ", got ", ctxSize);
        }

        hcclRet = HcclEngineCtxCopyFunc(commHandle, engine, mc2ContextTag.c_str(),
                                        &ctx, sizeof(Mc2MoeContext), CONTEXT_COPY_OFFSET);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Copy context to device failed, ret: ", hcclRet);
    }

    void GetHcclCommResource(const HcclComm &commHandle, const CommEngine &engine,
                             const CommProtocol &protocol, Mc2MoeContext &ctx,
                             uint32_t rankSize, uint64_t &hcclBuffSize) {
        uint32_t rankId = ctx.epRankId;
        uint32_t channelNum = rankSize - 1;

        uint32_t *netLayerList = nullptr;
        uint32_t netLayerNum = 0;
        GetNetLayers(commHandle, netLayerList, netLayerNum);
        TORCH_CHECK(netLayerNum > 0, "Get HCCL net layers failed, netLayerNum is ", netLayerNum);

        uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];
        uint32_t rankSizePerServer = 0;
        GetRankSizePerServer(commHandle, netLayers, rankSizePerServer);
        ctx.winSize = rankSizePerServer;

        std::vector<HcclChannelDesc> channelDesc(channelNum);
        auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HCCL channel init failed, ret: ", hcclRet);

        for (uint32_t i = 0; i < rankSize; ++i) {
            if (i == rankId) continue;
            uint32_t dstRank = i;
            uint32_t channelId = (i > rankId) ? (i - 1) : i;

            uint32_t layerId = (netLayerNum == HCCL_COMM_LAYERS_MTE_CCU) ?
                netLayerList[HCCL_COMM_LAYERS_UB_MEM] : layerMap_[dstRank];
            CommLink *links = nullptr;
            GetHcclCommLink(commHandle, layerId, rankId, dstRank, protocol, links);

            channelDesc[channelId].channelProtocol = protocol;
            channelDesc[channelId].remoteRank = dstRank;
            channelDesc[channelId].notifyNum = channelNum;
            channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
            channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
        }

        std::vector<ChannelHandle> channels(channelNum);
        hcclRet = HcclChannelAcquireFunc(commHandle, engine, channelDesc.data(), channelNum, channels.data());
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Acquire HCCL channel failed, ret: ", hcclRet);

        for (uint32_t i = 0; i < rankSize; ++i) {
            void *tempBuffer = nullptr;
            uint64_t bufSize = 0;

            if (i == rankId) {
                hcclRet = HcclGetHcclBufferFunc(commHandle, &tempBuffer, &hcclBuffSize);
            } else {
                uint32_t idx = (i < rankId) ? i : (i - 1);
                hcclRet = HcclChannelGetHcclBufferFunc(commHandle,
                    channels[idx], &tempBuffer, &bufSize);
            }
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL buffer failed src=", rankId,
                        " dst=", i, " ret=", hcclRet);
            ctx.epHcclBuffer[i] = reinterpret_cast<uint64_t>(tempBuffer);
        }
    }

    static void GetNetLayers(const HcclComm &commHandle, uint32_t *&netLayerList, uint32_t &netLayerNum) {
        auto hcclRet = HcclRankGraphGetLayersFunc(commHandle, &netLayerList, &netLayerNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", hcclRet);
    }

    static void GetRankSizePerServer(const HcclComm &commHandle, uint32_t netLayers,
                                      uint32_t &rankSizePerServer) {
        auto hcclRet = HcclRankGraphGetRankSizeByLayerFunc(commHandle, netLayers, &rankSizePerServer);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size per server failed, ret: ", hcclRet);
    }

    static void GetHcclCommLink(const HcclComm &commHandle, uint32_t netLayerId, uint32_t srcRankId,
                                uint32_t dstRankId, const CommProtocol &protocol, CommLink *&links) {
        CommLink *linksList = nullptr;
        uint32_t netLinkNum = 0;
        auto hcclRet = HcclRankGraphGetLinksFunc(commHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL communication link failed, ret: ", hcclRet);
        TORCH_CHECK(netLinkNum > 0, "No available HCCL links found. srcRankId: ", srcRankId,
                    ", dstRankId: ", dstRankId, ", layerId: ", netLayerId);

        for (uint32_t i = 0; i < netLinkNum; ++i) {
            if (linksList[i].linkAttr.linkProtocol == protocol) {
                links = &linksList[i];
                return;
            }
        }
        TORCH_CHECK(false, "No matching communication protocol found in HCCL links, protocol: ",
                    static_cast<int>(protocol));
    }

    std::unordered_map<uint32_t, uint32_t> layerMap_;
};

inline int64_t NormalizeDispatchQuantOutType(int64_t dispatchQuantOutType) {
    const int64_t aclE5m2 = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    const int64_t aclE4m3fn = static_cast<int64_t>(ACL_FLOAT8_E4M3FN);
    if (dispatchQuantOutType == TORCH_DTYPE_FLOAT8_E5M2 || dispatchQuantOutType == aclE5m2) {
        return aclE5m2;
    }
    if (dispatchQuantOutType == TORCH_DTYPE_FLOAT8_E4M3FN || dispatchQuantOutType == aclE4m3fn) {
        return aclE4m3fn;
    }
    TORCH_CHECK(false,
                "dispatch_quant_out_type only supports 23(torch.float8_e5m2), "
                "24(torch.float8_e4m3fn), ACL_FLOAT8_E5M2, or ACL_FLOAT8_E4M3FN, got ",
                dispatchQuantOutType);
}

inline bool IsSupportedMegaMoeHiddenSize(int64_t hiddenSize) {
    return hiddenSize == 4096 || hiddenSize == 5120 || hiddenSize == 7168;
}

inline bool IsSupportedMegaMoeIntermediateSize(int64_t intermediateSize) {
    return intermediateSize == 1024 || intermediateSize == 2048 ||
           intermediateSize == 3072 || intermediateSize == 4096 ||
           intermediateSize == 7168;
}

// ---- MegaMoe torch adapter ----
inline std::tuple<at::Tensor, at::Tensor> npu_mega_moe(
    const at::Tensor& x,
    const at::Tensor& topk_ids,
    const at::Tensor& topk_weights,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    c10::string_view group_ep,
    int64_t moe_expert_num,
    c10::optional<at::TensorList> weight_scales1 = c10::nullopt,
    c10::optional<at::TensorList> weight_scales2 = c10::nullopt,
    c10::optional<at::Tensor> x_active_mask = c10::nullopt,
    c10::optional<at::Tensor> scales = c10::nullopt,
    int64_t max_recv_token_num = 0,
    int64_t dispatch_quant_mode = DISPATCH_QUANT_MODE_MXFP,
    int64_t dispatch_quant_out_type = TORCH_DTYPE_FLOAT8_E4M3FN,
    int64_t combine_quant_mode = 0,
    c10::string_view comm_alg = "",
    int64_t global_bs = 0)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor, got dim: ", x.dim());
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be a 2D tensor, got dim: ", topk_ids.dim());
    TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be a 2D tensor, got dim: ", topk_weights.dim());
    TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(),
                "topk_ids and topk_weights must have the same shape, got ",
                topk_ids.sizes(), " and ", topk_weights.sizes());
    TORCH_CHECK(topk_ids.size(0) == x.size(0),
                "topk_ids batch size must match x, got ", topk_ids.size(0), " and ", x.size(0));
    TORCH_CHECK(topk_ids.scalar_type() == at::kInt,
                "topk_ids must be int32, got ", topk_ids.scalar_type());
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
                "x must be bfloat16 or float16, got ", x.scalar_type());

    const int64_t bs = x.size(0);
    const int64_t h = x.size(1);
    const int64_t topk = topk_ids.size(1);
    TORCH_CHECK(bs >= 1 && bs <= 512,
                "x dim0(BS) must be in [1, 512], got ", bs);
    TORCH_CHECK(IsSupportedMegaMoeHiddenSize(h),
                "x dim1(H) only supports 4096, 5120, or 7168, got ", h);
    TORCH_CHECK(topk == 6 || topk == 8,
                "topk_ids dim1(topK) only supports 6 or 8, got ", topk);

    TORCH_CHECK(weight1.size() > 0, "weight1 must not be empty");
    TORCH_CHECK(weight1.size() == weight2.size(),
                "weight1 and weight2 must have the same length, got ",
                weight1.size(), " and ", weight2.size());
    TORCH_CHECK(weight1[0].dim() == 3, "weight1[0] must be a 3D tensor, got dim: ", weight1[0].dim());
    TORCH_CHECK(weight2[0].dim() == 3, "weight2[0] must be a 3D tensor, got dim: ", weight2[0].dim());

    const int64_t expert_per_rank = weight1[0].size(0);
    const int64_t n = weight1[0].size(1);
    TORCH_CHECK(expert_per_rank >= 1 && expert_per_rank <= 16,
                "weight1[0] dim0(expertPerRank) must be in [1, 16], got ",
                expert_per_rank);
    TORCH_CHECK(IsSupportedMegaMoeIntermediateSize(n),
                "weight1[0] dim1(N) only supports 1024, 2048, 3072, 4096, or 7168, got ", n);
    TORCH_CHECK(weight1[0].size(2) == h,
                "weight1[0] dim2 must equal x dim1(H), got ",
                weight1[0].size(2), " and ", h);
    TORCH_CHECK(weight2[0].size(0) == expert_per_rank,
                "weight2[0] dim0 must equal expertPerRank, got ",
                weight2[0].size(0), " and ", expert_per_rank);
    TORCH_CHECK(weight2[0].size(1) == h,
                "weight2[0] dim1 must equal x dim1(H), got ",
                weight2[0].size(1), " and ", h);
    TORCH_CHECK(n == weight2[0].size(2) * 2,
                "weight1[0] dim1(N) must equal weight2[0] dim2 * 2, got N=",
                n, " and weight2[0] dim2=", weight2[0].size(2));

    TORCH_CHECK(dispatch_quant_mode == DISPATCH_QUANT_MODE_MXFP,
                "npu_mega_moe only supports dispatch_quant_mode=4(MXFP), got ",
                dispatch_quant_mode);
    TORCH_CHECK(combine_quant_mode == 0,
                "combine_quant_mode must be 0, got ", combine_quant_mode);
    TORCH_CHECK(comm_alg.empty(), "comm_alg must be an empty string, got ", std::string(comm_alg));
    TORCH_CHECK(!x_active_mask.has_value(),
                "x_active_mask must be None because current MegaMoe does not support non-empty x_active_mask");
    TORCH_CHECK(!scales.has_value(),
                "scales must be None because current MegaMoe does not support non-empty scales");
    TORCH_CHECK(dispatch_quant_out_type == TORCH_DTYPE_FLOAT8_E5M2 ||
                    dispatch_quant_out_type == TORCH_DTYPE_FLOAT8_E4M3FN,
                "dispatch_quant_out_type only supports 23(torch.float8_e5m2) "
                "or 24(torch.float8_e4m3fn), got ",
                dispatch_quant_out_type);
    NormalizeDispatchQuantOutType(dispatch_quant_out_type);

    MegaMoeContextBuilder builder;
    MegaMoeRuntimeContext runtime = builder.Build(group_ep);
    TORCH_CHECK(runtime.epWorldSize >= 2 && runtime.epWorldSize <= 768,
                "epWorldSize must be in [2, 768], got ", runtime.epWorldSize);
    TORCH_CHECK(moe_expert_num >= runtime.epWorldSize && moe_expert_num <= 1024,
                "moe_expert_num must be in [epWorldSize, 1024], got moe_expert_num=",
                moe_expert_num, ", epWorldSize=", runtime.epWorldSize);
    TORCH_CHECK(moe_expert_num % runtime.epWorldSize == 0,
                "moe_expert_num must be divisible by EP world size, got moe_expert_num=",
                moe_expert_num, ", ep_world_size=", runtime.epWorldSize);

    int64_t local_expert_num = moe_expert_num / runtime.epWorldSize;
    TORCH_CHECK(local_expert_num >= 1 && local_expert_num <= 16,
                "expertPerRank = moe_expert_num / epWorldSize must be in [1, 16], got ",
                local_expert_num);
    TORCH_CHECK(max_recv_token_num >= 0 &&
                    max_recv_token_num <= bs * runtime.epWorldSize * std::min(topk, local_expert_num),
                "max_recv_token_num must be in [0, BS * epWorldSize * min(topK, expertPerRank)], got ",
                max_recv_token_num, ", upper bound is ",
                bs * runtime.epWorldSize * std::min(topk, local_expert_num));
    TORCH_CHECK(global_bs == 0 ||
                    (global_bs >= bs * runtime.epWorldSize && global_bs % runtime.epWorldSize == 0),
                "global_bs must be 0 or satisfy BS * epWorldSize <= global_bs and "
                "global_bs % epWorldSize == 0, got global_bs=",
                global_bs, ", BS=", bs, ", epWorldSize=", runtime.epWorldSize);
    TORCH_CHECK(weight1[0].size(0) == local_expert_num,
                "weight1[0] dim0 must equal local expert num, got ",
                weight1[0].size(0), " and expected ", local_expert_num);
    TORCH_CHECK(weight2[0].size(0) == local_expert_num,
                "weight2[0] dim0 must equal local expert num, got ",
                weight2[0].size(0), " and expected ", local_expert_num);
    TORCH_CHECK(weight_scales1.has_value() && weight_scales1.value().size() > 0,
                "weight_scales1 must not be None or empty");
    TORCH_CHECK(weight_scales2.has_value() && weight_scales2.value().size() > 0,
                "weight_scales2 must not be None or empty");
    TORCH_CHECK(weight_scales1.value()[0].dim() == 4,
                "weight_scales1[0] must be a 4D tensor, got dim: ",
                weight_scales1.value()[0].dim());
    TORCH_CHECK(weight_scales2.value()[0].dim() == 4,
                "weight_scales2[0] must be a 4D tensor, got dim: ",
                weight_scales2.value()[0].dim());
    TORCH_CHECK(weight_scales1.value()[0].size(0) == local_expert_num,
                "weight_scales1[0] dim0 must equal local expert num, got ",
                weight_scales1.value()[0].size(0), " and expected ", local_expert_num);
    TORCH_CHECK(weight_scales2.value()[0].size(0) == local_expert_num,
                "weight_scales2[0] dim0 must equal local expert num, got ",
                weight_scales2.value()[0].size(0), " and expected ", local_expert_num);
    TORCH_CHECK(weight_scales1.value()[0].size(1) == n,
                "weight_scales1[0] dim1 must equal N, got ",
                weight_scales1.value()[0].size(1), " and ", n);
    TORCH_CHECK(weight_scales1.value()[0].size(2) == (h + 63) / 64,
                "weight_scales1[0] dim2 must equal ceil(H / 64), got ",
                weight_scales1.value()[0].size(2), " and expected ", (h + 63) / 64);
    TORCH_CHECK(weight_scales1.value()[0].size(3) == 2,
                "weight_scales1[0] dim3 must be 2, got ", weight_scales1.value()[0].size(3));
    TORCH_CHECK(weight_scales2.value()[0].size(1) == h,
                "weight_scales2[0] dim1 must equal H, got ",
                weight_scales2.value()[0].size(1), " and ", h);
    TORCH_CHECK(weight_scales2.value()[0].size(2) == ((n / 2) + 63) / 64,
                "weight_scales2[0] dim2 must equal ceil((N / 2) / 64), got ",
                weight_scales2.value()[0].size(2), " and expected ", ((n / 2) + 63) / 64);
    TORCH_CHECK(weight_scales2.value()[0].size(3) == 2,
                "weight_scales2[0] dim3 must be 2, got ", weight_scales2.value()[0].size(3));

    at::Tensor y = at::empty({bs, h}, x.options());
    at::Tensor expert_token_nums = at::empty({local_expert_num},
        x.options().dtype(at::kInt));

    // 3. Prepare optional tensor lists
    at::TensorList w1_scales = weight_scales1.has_value() ? weight_scales1.value() : at::TensorList();
    at::TensorList w2_scales = weight_scales2.has_value() ? weight_scales2.value() : at::TensorList();
    at::Tensor x_mask = x_active_mask.has_value() ? x_active_mask.value() : at::Tensor();
    at::Tensor s = scales.has_value() ? scales.value() : at::Tensor();

    int64_t dispatch_quant_out_type_acl = NormalizeDispatchQuantOutType(dispatch_quant_out_type);
    std::string comm_alg_storage(comm_alg);
    char* comm_alg_ptr = const_cast<char*>(comm_alg_storage.c_str());

    EXEC_NPU_CMD(aclnnMegaMoe,
        runtime.context, x, topk_ids, topk_weights,
        weight1, weight2,
        w1_scales, w2_scales,
        x_mask, s,
        moe_expert_num,
        runtime.epWorldSize,
        runtime.cclBufferSize,
        max_recv_token_num, dispatch_quant_mode, dispatch_quant_out_type_acl,
        combine_quant_mode, comm_alg_ptr, global_bs,
        y, expert_token_nums);

    return {y, expert_token_nums};
}

} // namespace vllm_ascend
#endif
