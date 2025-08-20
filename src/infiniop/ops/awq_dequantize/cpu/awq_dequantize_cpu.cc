#include "awq_dequantize_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <cstdint>
#include <cmath>

namespace op::awq_dequantize::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t qweight_desc,
    infiniopTensorDescriptor_t zeros_desc,
    infiniopTensorDescriptor_t scales_desc,
    int group_size) {
    auto result = AWQDequantizeInfo::create(y_desc, qweight_desc, zeros_desc, scales_desc, group_size);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// AWQ 重新排序模式
constexpr int AWQ_REVERSE_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

// 解包和重新排序函数
template <typename T>
void unpack_and_reorder(T* unpacked, const int32_t* packed, int n, int m_packed, int m) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m_packed; j++) {
            int32_t packed_val = packed[i * m_packed + j];
            
            // 解包并重新排序
            for (int k = 0; k < 8; k++) {
                int target_pos = j * 8 + AWQ_REVERSE_ORDER[k];
                T value = (packed_val >> (k * 4)) & 0xF;
                unpacked[i * m + target_pos] = value;
            }
        }
    }
}

// 主反量化函数
template <typename Tdata, typename Tscale>
infiniStatus_t awq_dequantize(
    const AWQDequantizeInfo *info,
    Tdata *y,
    const int32_t *qweight,
    const int8_t *zeros,
    const Tscale *scales) {
    
    int n = info->n;
    int m = info->m;
    int m_packed = info->m_packed;
    int group_size = info->group_size;
    int zeros_n = info->zeros_n;
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int group_idx = i / group_size;
        
        for (int j = 0; j < m; j++) {
            // 计算权重在打包数组中的位置
            int packed_col = j / 8;
            int bit_shift = (j % 8) * 4;
            
            // 读取打包的权重值
            int32_t packed_weight = qweight[i * m_packed + packed_col];
            
            // 提取4位权重值并应用重新排序
            int rev_index = AWQ_REVERSE_ORDER[j % 8];
            int8_t weight_val = (packed_weight >> (rev_index * 4)) & 0xF;
            
            // 读取对应的零点值和缩放因子
            int8_t zero_val = zeros[group_idx * m + j];
            Tscale scale_val = scales[group_idx * m + j];
            
            // 计算反量化值
            y[i * m + j] = static_cast<Tdata>((weight_val - zero_val) * scale_val);
        }
    }
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *qweight, const void *zeros, const void *scales,
    void *stream) const {
    
    // 分配临时内存用于解包后的零点
    int zeros_size = _info.zeros_n * _info.zeros_m;
    int8_t* unpacked_zeros = static_cast<int8_t*>(workspace);
    
    // 解包和重新排序零点
    unpack_and_reorder(unpacked_zeros, 
                      static_cast<const int32_t*>(zeros), 
                      _info.zeros_n, 
                      _info.zeros_m_packed, 
                      _info.zeros_m);
    
    // 执行反量化
    if (_info.data_type == INFINI_DTYPE_F16) {
        if (_info.scale_type == INFINI_DTYPE_F16) {
            CHECK_STATUS(awq_dequantize(&_info, 
                                       static_cast<fp16_t*>(y), 
                                       static_cast<const int32_t*>(qweight), 
                                       unpacked_zeros, 
                                       static_cast<const fp16_t*>(scales)));
        } else if (_info.scale_type == INFINI_DTYPE_F32) {
            CHECK_STATUS(awq_dequantize(&_info, 
                                       static_cast<fp16_t*>(y), 
                                       static_cast<const int32_t*>(qweight), 
                                       unpacked_zeros, 
                                       static_cast<const float*>(scales)));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_info.data_type == INFINI_DTYPE_F32) {
        if (_info.scale_type == INFINI_DTYPE_F16) {
            CHECK_STATUS(awq_dequantize(&_info, 
                                       static_cast<float*>(y), 
                                       static_cast<const int32_t*>(qweight), 
                                       unpacked_zeros, 
                                       static_cast<const fp16_t*>(scales)));
        } else if (_info.scale_type == INFINI_DTYPE_F32) {
            CHECK_STATUS(awq_dequantize(&_info, 
                                       static_cast<float*>(y), 
                                       static_cast<const int32_t*>(qweight), 
                                       unpacked_zeros, 
                                       static_cast<const float*>(scales)));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::awq_dequantize::cpu