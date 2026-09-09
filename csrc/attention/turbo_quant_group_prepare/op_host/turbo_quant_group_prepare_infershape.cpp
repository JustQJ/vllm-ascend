// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
#include "../../turbo_quant_sparse_flash_attention/turbo_quant_group_plan.h"
namespace ops {
static ge::graphStatus InferGroupPrepare(gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(0);
    if (input == nullptr || input->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < 3; ++i) {
        auto *out = context->GetOutputShape(i);
        if (out == nullptr) {
            return ge::GRAPH_FAILED;
        }
        out->SetDimNum(2);
        out->SetDim(0, input->GetDim(0));
        out->SetDim(1, i == 0 ? tq_group::DESC_WORDS : input->GetDim(2));
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferGroupPrepareType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_UINT8);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(TurboQuantGroupPrepare).InferShape(InferGroupPrepare).InferDataType(InferGroupPrepareType);
} // namespace ops
