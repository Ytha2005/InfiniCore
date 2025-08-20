#ifndef AWQ_DEQUANTIZE_H
#define AWQ_DEQUANTIZE_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::awq_dequantize::NAMESPACE {                    \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        AWQDequantizeInfo _info;                                 \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            AWQDequantizeInfo info,                              \
            size_t workspace_size,                               \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t y_desc,                   \
            infiniopTensorDescriptor_t qweight_desc,             \
            infiniopTensorDescriptor_t zeros_desc,               \
            infiniopTensorDescriptor_t scales_desc,              \
            int group_size);                                     \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *y,                                             \
            const void *qweight,                                 \
            const void *zeros,                                   \
            const void *scales,                                  \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // AWQ_DEQUANTIZE_H