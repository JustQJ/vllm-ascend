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
#ifndef MEGA_MOE_TORCH_ADPT_H
#define MEGA_MOE_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_mega_moe(
    const at::Tensor &context,
    const at::Tensor &x,
    const at::Tensor &topk_ids,
    const at::Tensor &topk_weights,
    const at::TensorList &weight1,
    const at::TensorList &weight2,
    int64_t moe_expert_num,
    int64_t ep_world_size,
    int64_t ccl_buffer_size,
    const c10::optional<at::TensorList> &weight_scales1,
    const c10::optional<at::TensorList> &weight_scales2,
    const c10::optional<at::TensorList> &bias1,
    const c10::optional<at::TensorList> &bias2,
    const c10::optional<at::Tensor> &x_active_mask,
    int64_t max_recv_token_num,
    int64_t dispatch_quant_mode,
    int64_t combine_quant_mode,
    c10::string_view comm_alg,
    int64_t num_max_tokens_per_rank,
    c10::string_view activation,
    c10::optional<double> activation_clamp,
    c10::optional<int64_t> dispatch_quant_out_dtype,
    c10::optional<int64_t> weight1_type,
    c10::optional<int64_t> weight2_type)
{
    // Input validation
    TORCH_CHECK((ep_world_size > 0),
        "The ep_world_size should be greater than 0, current is: ", ep_world_size);
    TORCH_CHECK((x.dim() == 2) && (topk_ids.dim() == 2),
        "The x and topk_ids should be 2D");
    TORCH_CHECK(((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf)) &&
                (topk_ids.scalar_type() == at::kInt),
        "dtype of x should be bfloat16, float16, dtype of topk_ids should be int.");

    // Resolve optional tensor lists into concrete references
    at::TensorList weight_scales1_ref;
    if (weight_scales1.has_value()) {
        weight_scales1_ref = weight_scales1.value();
    } else {
        weight_scales1_ref = at::TensorList();
    }
    at::TensorList weight_scales2_ref;
    if (weight_scales2.has_value()) {
        weight_scales2_ref = weight_scales2.value();
    } else {
        weight_scales2_ref = at::TensorList();
    }
    at::TensorList bias1_ref;
    if (bias1.has_value()) {
        bias1_ref = bias1.value();
    } else {
        bias1_ref = at::TensorList();
    }
    at::TensorList bias2_ref;
    if (bias2.has_value()) {
        bias2_ref = bias2.value();
    } else {
        bias2_ref = at::TensorList();
    }

    // Determine ACL data types for weights.
    // weight1_type / weight2_type override inference from tensor scalar type.
    aclDataType weight1_ref_dtype = weight1_type.has_value()
        ? static_cast<aclDataType>(weight1_type.value())
        : ConvertType(weight1[0].scalar_type());
    aclDataType weight_scales1_dtype;
    if (weight1_ref_dtype == aclDataType::ACL_FLOAT8_E5M2 ||
        weight1_ref_dtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight1_ref_dtype == aclDataType::ACL_FLOAT4_E2M1) {
        weight_scales1_dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weight_scales1_dtype = aclDataType::ACL_UINT64;
    }

    aclDataType weight2_ref_dtype = weight2_type.has_value()
        ? static_cast<aclDataType>(weight2_type.value())
        : ConvertType(weight2[0].scalar_type());
    aclDataType weight_scales2_dtype;
    if (weight2_ref_dtype == aclDataType::ACL_FLOAT8_E5M2 ||
        weight2_ref_dtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight2_ref_dtype == aclDataType::ACL_FLOAT4_E2M1) {
        weight_scales2_dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weight_scales2_dtype = aclDataType::ACL_UINT64;
    }

    // Compute output shapes
    auto x_shape = x.sizes();
    int64_t bs = x_shape[0];
    int64_t h = x_shape[1];
    int64_t num_local_experts = moe_expert_num / ep_world_size;

    // Validate dispatch_quant_out_dtype for FLOAT4_E2M1
    if ((dispatch_quant_out_dtype.has_value()) &&
        (dispatch_quant_out_dtype.value() == static_cast<int64_t>(aclDataType::ACL_FLOAT4_E2M1))) {
        TORCH_CHECK(h % 2 == 0,
            "The last dim input shape must be divisible by 2 if "
            "dispatch quant output type is float4_e2m1");
    }

    // Allocate outputs
    at::Tensor y_out = at::empty({bs, h}, x.options());
    auto opts = x.options().dtype(at::kInt);
    at::Tensor expert_token_nums_out = at::empty({num_local_experts}, opts);

    // Resolve dispatch_quant_result_type
    int64_t dispatch_quant_result_type = dispatch_quant_out_dtype.has_value()
        ? static_cast<int64_t>(static_cast<aclDataType>(dispatch_quant_out_dtype.value()))
        : 28;  // ge::DT_UNDEFINED

    // Resolve float and string params
    float activation_clamp_f = activation_clamp.has_value()
        ? static_cast<float>(activation_clamp.value())
        : std::numeric_limits<float>::max();

    std::string comm_alg_str(comm_alg);
    char *comm_alg_ptr = comm_alg_str.data();
    std::string activation_str(activation);
    char *activation_ptr = activation_str.data();

    // Wrap tensor lists with explicit ACL dtypes for the ACLNN operator
    TensorListWrapper weight1_wrapper = {weight1, weight1_ref_dtype};
    TensorListWrapper weight2_wrapper = {weight2, weight2_ref_dtype};
    TensorListWrapper weight_scales1_wrapper = {weight_scales1_ref, weight_scales1_dtype};
    TensorListWrapper weight_scales2_wrapper = {weight_scales2_ref, weight_scales2_dtype};
    TensorListWrapper bias1_wrapper = {bias1_ref, aclDataType::ACL_FLOAT};
    TensorListWrapper bias2_wrapper = {bias2_ref, aclDataType::ACL_FLOAT};

    EXEC_NPU_CMD(aclnnMegaMoe,
        context, x, topk_ids, topk_weights,
        weight1_wrapper, weight2_wrapper,
        weight_scales1_wrapper, weight_scales2_wrapper,
        bias1_wrapper, bias2_wrapper,
        x_active_mask,
        moe_expert_num, ep_world_size, ccl_buffer_size,
        max_recv_token_num, dispatch_quant_mode, dispatch_quant_result_type,
        combine_quant_mode, comm_alg_ptr,
        num_max_tokens_per_rank, activation_ptr, activation_clamp_f,
        y_out, expert_token_nums_out);

    return {y_out, expert_token_nums_out};
}

}  // namespace vllm_ascend
#endif
