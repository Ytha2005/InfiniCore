#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::awq_dequantize {
struct Test::Attributes {
    int group_size;
    std::shared_ptr<Tensor> qweight;
    std::shared_ptr<Tensor> zeros;
    std::shared_ptr<Tensor> scales;
    std::shared_ptr<Tensor> ans;
    std::shared_ptr<Tensor> y;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (attributes.find("group_size") == attributes.end()
        || tensors.find("qweight") == tensors.end()
        || tensors.find("zeros") == tensors.end()
        || tensors.find("scales") == tensors.end()
        || tensors.find("ans") == tensors.end()
        || tensors.find("y") == tensors.end()) {
        throw std::runtime_error("Invalid Test: Missing attributes or tensors");
    }

    test->_attributes->group_size = *reinterpret_cast<int *>(attributes["group_size"].data());

    test->_attributes->qweight = tensors["qweight"];
    test->_attributes->zeros = tensors["zeros"];
    test->_attributes->scales = tensors["scales"];
    test->_attributes->ans = tensors["ans"];
    test->_attributes->y = tensors["y"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopAWQDequantizeDescriptor_t op_desc;
    CHECK_OR(infiniopCreateAWQDequantizeDescriptor(handle, &op_desc,
                                                  _attributes->y->desc(),
                                                  _attributes->qweight->desc(),
                                                  _attributes->zeros->desc(),
                                                  _attributes->scales->desc(),
                                                  _attributes->group_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create AWQDequantize descriptor"));

    auto qweight = _attributes->qweight->to(device, device_id);
    auto zeros = _attributes->zeros->to(device, device_id);
    auto scales = _attributes->scales->to(device, device_id);
    auto y = _attributes->y->to(device, device_id);

    size_t workspace_size;
    CHECK_OR(infiniopGetAWQDequantizeWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size"));
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace"));
    }

    CHECK_OR(infiniopAWQDequantize(op_desc,
                                  workspace, workspace_size,
                                  y->data(),
                                  qweight->data(),
                                  zeros->data(),
                                  scales->data(),
                                  nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "AWQDequantize execution failed"));

    try {
        allClose(y, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopAWQDequantize(op_desc,
                                 workspace, workspace_size,
                                 y->data(),
                                 qweight->data(),
                                 zeros->data(),
                                 scales->data(),
                                 nullptr);
        },
        warm_ups, iterations);

    if (workspace != nullptr) {
        infinirtFree(workspace);
    }

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"group_size"};
}

std::vector<std::string> Test::tensor_names() {
    return {"qweight", "zeros", "scales", "ans", "y"};
}

std::vector<std::string> Test::output_names() {
    return {"y"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- group_size=" << _attributes->group_size << std::endl;
    oss << "- qweight: " << _attributes->qweight->info() << std::endl;
    oss << "- zeros: " << _attributes->zeros->info() << std::endl;
    oss << "- scales: " << _attributes->scales->info() << std::endl;
    oss << "- y: " << _attributes->y->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::awq_dequantize