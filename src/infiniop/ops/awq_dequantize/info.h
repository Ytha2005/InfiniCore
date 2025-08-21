#ifndef __AWQ_DEQUANTIZE_INFO_H__
#define __AWQ_DEQUANTIZE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::awq_dequantize {

class AWQDequantizeInfo {
    AWQDequantizeInfo() = default;

public:
    infiniDtype_t zero_type;     // 零点数据类型 (通常是 INT8)
    infiniDtype_t scale_type;    // 缩放因子数据类型 (F16/F32)
    infiniDtype_t data_type;     // 输出数据类型 (F16/F32)
    int group_size;              // 分组大小
    int n;                       // 行数
    int m;                       // 列数 (解包后的)
    int m_packed;                // 列数 (打包后的)
    int zeros_n;                 // 零点行数
    int zeros_m;                 // 零点列数 (解包后的)
    int zeros_m_packed;          // 零点列数 (打包后的)

    static utils::Result<AWQDequantizeInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t qweight_desc,
        infiniopTensorDescriptor_t zeros_desc,
        infiniopTensorDescriptor_t scales_desc,
        int group_size) {

        // 检查数据类型
        auto data_type = y_desc->dtype();
        auto zero_type = zeros_desc->dtype();
        auto scale_type = scales_desc->dtype();
        auto qweight_type = qweight_desc->dtype();

        // 量化权重必须是 INT32
        if (qweight_type != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 零点必须是 INT32
        if (zero_type != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 输出和缩放因子必须是浮点类型
        if (data_type != INFINI_DTYPE_F16 && data_type != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (scale_type != INFINI_DTYPE_F16 && scale_type != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 检查维度
        if (y_desc->ndim() != 2 || qweight_desc->ndim() != 2 || 
            zeros_desc->ndim() != 2 || scales_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 获取维度信息
        int n = y_desc->shape()[0];
        int m = y_desc->shape()[1];
        int zeros_n = zeros_desc->shape()[0];
        int zeros_m = scales_desc->shape()[1];  // 解包后的零点列数

        // 计算打包后的维度
        int m_packed = m / 8;
        int zeros_m_packed = zeros_m / 8;

        return utils::Result<AWQDequantizeInfo>(AWQDequantizeInfo{
            zero_type,
            scale_type,
            data_type,
            group_size,
            n,
            m,
            m_packed,
            zeros_n,
            zeros_m,
            zeros_m_packed
        });
    }
};

} // namespace op::awq_dequantize

#endif // __AWQ_DEQUANTIZE_INFO_H__