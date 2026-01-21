#include "operators/matmul.h"

namespace infini {

/**
 * @brief 构造矩阵乘法操作符对象
 * @param graph 计算图指针，该操作符将被添加到此图中
 * @param A 输入张量 A，参与矩阵乘法的第一个矩阵
 * @param B 输入张量 B，参与矩阵乘法的第二个矩阵
 * @param C 输出张量 C，存储矩阵乘法的计算结果
 * @param transA 是否对张量 A 进行转置，true 表示转置，false 表示不转置
 * @param transB 是否对张量 B 进行转置，true 表示转置，false 表示不转置
 *
 * 矩阵乘法运算说明：
 * - 当 transA=false 且 transB=false 时，计算 C = A × B
 * - 当 transA=true 且 transB=false 时，计算 C = A^T × B
 * - 当 transA=false 且 transB=true 时，计算 C = A × B^T
 * - 当 transA=true 且 transB=true 时，计算 C = A^T × B^T
 */
MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
      transA(transA), transB(transB) {
    // 检查操作符配置是否有效，包括输入输出张量的维度是否兼容
    IT_ASSERT(checkValid(graph));
}

/**
 * @brief 将矩阵乘法操作符转换为可读的字符串描述
 * @return 包含操作符详细信息的字符串
 *
 * 返回的字符串包含以下信息：
 * 1. 操作类型和是否对输入矩阵进行转置
 * 2. 输入张量 A 和 B 的唯一标识符
 * 3. 输出张量 C 的唯一标识符
 * 4. 矩阵乘法的维度信息 [m, n, k]，其中：
 *    - m 是结果矩阵的行数
 *    - n 是结果矩阵的列数
 *    - k 是矩阵乘法的内积维度（点积长度）
 */
string MatmulObj::toString() const {
    std::ostringstream os;
    // 构建描述字符串，显示输入矩阵的转置状态和维度信息
    os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
       << ",A=" << inputs[0]->getGuid()
       << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
       << ",mnk=[" << m << "," << n << "," << k << "])";
    return os.str();
}

/**
 * @brief 推断矩阵乘法操作后输出张量的形状
 * @param inputs 输入张量向量，包含两个矩阵 A 和 B
 * @return 输出张量的形状向量，如果推断失败返回 std::nullopt
 *
 * 矩阵乘法的形状推断规则（遵循 ONNX GEMM 算子规范）：
 *
 * 对于二维矩阵：
 * - 如果 A 的形状为 [M, K]，B 的形状为 [K, N]，则 C 的形状为 [M, N]
 * - 支持 batch 模式的矩阵乘法
 *
 * 转置的影响：
 * - transA=true: A^T 对应最后一维和倒数第二维交换
 * - transB=true: B^T 对应最后一维和倒数第二维交换
 *
 * 广播规则：
 * - 输入张量支持广播机制
 * - 当维度不匹配时，自动进行广播以兼容矩阵乘法
 *
 * @note 这是需要实现的作业部分，参考 ONNX 官方文档：
 *       https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
 */
optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
    // =================================== 作业 ===================================
    // TODO：返回经过 matmul 操作后的 shape
    // 提示：
    // 1. 获取输入张量 A 和 B 的形状
    // 2. 根据 transA 和 transB 标志调整形状（转置会交换最后两维）
    // 3. 检查矩阵乘法的维度是否兼容（A 的最后一维度必须等于 B 的倒数第二维度）
    // 4. 根据矩阵乘法规则计算输出形状：
    //    - 输出形状的前缀维度由广播决定
    //    - 输出形状的最后两维为 [A 的倒数第二维度, B 的最后一维度]
    // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
    // =================================== 作业 ===================================

    // 1. 获取输入形状
    Shape shapeA = inputs[0]->getDims();
    Shape shapeB = inputs[1]->getDims();

    // 2. 处理转置
    if (transA) {
        if (shapeA.size() < 2)
            return std::nullopt;
        std::swap(shapeA[shapeA.size() - 2], shapeA[shapeA.size() - 1]);
    }
    if (transB) {
        if (shapeB.size() < 2)
            return std::nullopt;
        std::swap(shapeB[shapeB.size() - 2], shapeB[shapeB.size() - 1]);
    }

    // 3. 检查维度兼容性
    if (shapeA.back() != shapeB[shapeB.size() - 2]) {
        return std::nullopt;
    }

    // 4. 提取前缀维度（去掉最后两维）
    Shape prefixA(shapeA.begin(), shapeA.end() - 2);

    // 5. 计算输出形状
    Shape outputShape = prefixA;
    outputShape.push_back(shapeA[shapeA.size() - 2]);
    outputShape.push_back(shapeB[shapeB.size() - 1]);

    return {{outputShape}};
}

} // namespace infini