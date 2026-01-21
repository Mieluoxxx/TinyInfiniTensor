/**
 * @file concat.cc
 * @brief Concat（拼接）操作符的实现
 *
 * 该文件实现了 Concat 操作符，用于在指定维度上拼接多个张量。
 * Concat 是深度学习中的基础操作，常用于将特征图或张量沿某一维度连接。
 *
 * 主要功能：
 * - 支持在任意维度上进行拼接（通过负数索引指定维度）
 * - 验证输入张量的兼容性（非拼接维度必须相同）
 * - 推断输出张量的形状
 */

#include "operators/concat.h"     // Concat 操作符的头文件定义
#include "utils/operator_utils.h" // 操作符工具函数，如 get_real_axis 等

namespace infini {

// ==================== ConcatObj 类实现 ====================
// Concat 操作符：在指定的维度上拼接多个张量

// ==================== ConcatObj 类实现 ====================
// Concat 操作符：在指定的维度上拼接多个张量

/**
 * @brief ConcatObj 构造函数
 * @param graph 计算图对象指针
 * @param inputs 输入张量向量（可以是多个张量）
 * @param output 输出张量
 * @param _dim 拼接的维度（可以是负数，表示从后往前数）
 *
 * Concat 操作将多个张量在指定的维度上进行拼接
 * 注意：除了拼接的维度外，其他维度的尺寸必须相同
 *
 * 示例：
 *   - 在 dim=0 上拼接 [[1,2], [3,4]] 和 [[5,6], [7,8]] → [[1,2], [3,4], [5,6], [7,8]]
 *   - 在 dim=1 上拼接 [[1,2], [3,4]] 和 [[5,6], [7,8]] → [[1,2,5,6], [3,4,7,8]]
 */
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    // 获取输入张量的秩（维度数量）
    int rank = inputs[0]->getRank();
    // 将用户提供的维度（可能是负数）转换为实际的轴索引
    // 例如：如果 rank=3，用户输入 _dim=-1，则实际 dim=2（最后一个维度）
    dim = get_real_axis(_dim, rank);
    // 验证操作符在计算图中的有效性
    IT_ASSERT(checkValid(graph));
}

/**
 * @brief 推断 Concat 操作后输出张量的形状
 * @param inputs 输入张量向量，包含需要拼接的所有张量
 * @return 包含输出形状的 vector（optional，可能为空表示推断失败）
 *
 * 该函数根据输入张量的形状拼接维度，计算输出张量的形状。
 *
 * 关键规则：
 * 1. 所有输入张量的秩（维度数量）必须相同
 * 2. 除了拼接维度外，其他维度的大小必须相同
 * 3. 沿拼接维度的输出大小为所有输入张量在该维度大小之和
 *
 * 例如：
 *   输入1: shape [2, 3, 4], 输入2: shape [2, 5, 4], 在 dim=1 上拼接
 *   输出: shape [2, 3+5, 4] = [2, 8, 4]
 */
optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    // 获取第一个输入张量的形状作为输出的初始形状
    // 假设第一个输入的维度是正确的基准
    Shape dims = inputs[0]->getDims();

    // 获取张量的秩（维度数量），用于后续验证和计算
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    
    // 验证所有输入张量的秩是否相同
    for (const auto &input : inputs) {
        if (input->getRank() != rank) {
            return std::nullopt;
        }
    }


    // 验证除了拼接维度外，其他维度的尺寸是否相同
    for (size_t i = 1; i < inputs.size(); ++i) {
        const auto &inputDims = inputs[i]->getDims();
        for (size_t j = 0; j < rank; ++j) {
            if (j != static_cast<size_t>(dim) && inputDims[j] != dims[j]) {
                return std::nullopt;
            }
        }
    }

    // 先将拼接维度清零，避免第一个输入被重复计算
    dims[dim] = 0;

    // 累加所有输入张量在拼接维度上的大小
    for (const auto &input : inputs) {
        dims[dim] += input->getDims()[dim];
    }

    return {{dims}};
}

/**
 * @brief 生成 Concat 操作符的字符串表示
 * @return 描述该 Concat 操作符的字符串
 *
 * 该方法用于调试和日志记录，将 Concat 操作符的关键信息格式化为可读字符串。
 * 输出格式示例：
 *   "Concat[guid-123]([2,3],[2,4],dim=1,input=guid-A,guid-B,output=guid-C)"
 *
 * 输出信息包括：
 * - 操作符类型和唯一标识符 (guid)
 * - 所有输入张量的形状
 * - 拼接的维度索引
 * - 输入张量的 guid 列表
 * - 输出张量的 guid
 */
std::string ConcatObj::toString() const {
    std::ostringstream os;

    // 输出操作符类型和唯一标识符
    os << "Concat[" << getGuid() << "]";

    // 输出左括号，开始参数列表
    os << "(";

    // 输出所有输入张量的形状
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";

    // 输出拼接的维度
    os << "dim=" << dim << ",";

    // 输出所有输入张量的 guid 标识符
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";

    // 输出输出张量的 guid
    os << "output=" << outputs[0]->getGuid() << ")";

    return os.str();
}

} // namespace infini
