#include "op/org.openvinotoolkit/celu.hpp"

#include <memory>

#include "default_opset.hpp"
#include "utils/common.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector celu(const Node& node) {
    const auto in1 = node.get_ng_inputs().at(0);
    const auto in2 = node.get_ng_inputs().at(1);
    const auto alpha = node.get_attribute_value<float>("alpha", 1);
    const auto alpha_node =
        std::make_shared<default_opset::Convert>(default_opset::Constant::create(element::f32, {}, {alpha}),
                                                 in1.get_element_type());

    const auto add = std::make_shared<default_opset::Add>(in1, in2);
    return {std::make_shared<default_opset::Multiply>(add, alpha_node)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph