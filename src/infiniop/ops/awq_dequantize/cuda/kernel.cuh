#ifndef __AWQ_DEQUANTIZE_CUDA_KERNEL_H__
#define __AWQ_DEQUANTIZE_CUDA_KERNEL_H__

#include <cuda_fp16.h>

// 预定义的重新排序模式
constexpr int AWQ_REVERSE_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

// 解包和重新排序零点值的核函数
template <unsigned int BLOCK_SIZE, typename Tzero>
__device__ void unpackZerosBlock(
    Tzero *__restrict__ unpacked_zeros,  // 输出: 解包后的零点值 [zeros_n, zeros_m]
    const int32_t *__restrict__ qzeros,  // 输入: 打包的零点值 [zeros_n, zeros_m_packed]
    int zeros_n,                         // 零点行数
    int zeros_m_packed,                  // 零点列数（打包后的）
    int zeros_m) {                       // 零点列数（解包后的）
    
    // 每个线程处理一个打包的零点元素
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= zeros_n || col >= zeros_m_packed) return;
    
    // 读取打包的int32值
    int32_t packed = qzeros[row * zeros_m_packed + col];
    
    // 解包并重新排序
    for (int i = 0; i < 8; i++) {
        int target_pos = col * 8 + AWQ_REVERSE_ORDER[i];
        Tzero value = (packed >> (i * 4)) & 0xF;
        unpacked_zeros[row * zeros_m + target_pos] = value;
    }
}

// 主反量化核函数
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tzero, typename Tscale>
__device__ void awqDequantizeBlock(
    Tdata *__restrict__ y,               // 输出: 反量化后的权重 [n, m]
    const int32_t *__restrict__ qweight, // 输入: 打包的权重 [n, m_packed]
    const Tzero *__restrict__ zeros,     // 输入: 解包后的零点值 [zeros_n, m]
    const Tscale *__restrict__ scales,   // 输入: 缩放因子 [zeros_n, m]
    int n,                               // 行数
    int m_packed,                        // 列数（打包后的）
    int m,                               // 列数（解包后的）
    int group_size) {                    // 分组大小
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= n || col >= m) return;
    
    // 计算组索引
    int group_idx = row / group_size;
    
    // 计算权重在打包数组中的位置
    int packed_col = col / 8;
    int bit_shift = (col % 8) * 4;
    
    // 读取打包的权重值
    int32_t packed_weight = qweight[row * m_packed + packed_col];
    
    // 提取4位权重值并应用重新排序
    int rev_index = AWQ_REVERSE_ORDER[col % 8];
    Tzero weight_val = (packed_weight >> (rev_index * 4)) & 0xF;
    
    // 读取对应的零点值和缩放因子
    Tzero zero_val = zeros[group_idx * m + col];
    Tscale scale_val = scales[group_idx * m + col];
    
    // 计算反量化值
    y[row * m + col] = Tdata((weight_val - zero_val) * scale_val);
}

#endif  // __AWQ_DEQUANTIZE_CUDA_KERNEL_H__