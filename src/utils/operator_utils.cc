#include "utils/operator_utils.h"
#include "core/common.h"
#include "core/runtime.h"

namespace infini {

/**
 * @brief 推断两个张量进行广播操作后的输出形状
 *
 * 广播机制允许不同形状的张量进行算术运算。广播遵循以下规则：
 * 1. 从右向左比较两个形状的各个维度
 * 2. 每个维度必须相等，或者其中一个是1，或者其中一个不存在
 * 3. 输出形状的每个维度是输入形状对应维度的最大值
 *
 * 例如：
 * - Shape A: (1, 3, 1, 5)
 * - Shape B: (3, 1, 5)
 * - 输出: (1, 3, 1, 5)
 *
 * @param A 第一个张量的形状
 * @param B 第二个张量的形状
 * @return 广播后的输出形状
 */
Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================

    // 1. 确定输出形状的维度数（取 A 和 B 中较大的）
    size_t maxDim = std::max(A.size(), B.size());
    Shape output(maxDim);

    // 2. 从右向左遍历每个维度
    for (int i = 0; i < static_cast<int>(maxDim); ++i) {
        // 计算 A 和 B 对应维度的索引
        int idxA = static_cast<int>(A.size()) - 1 - i;
        int idxB = static_cast<int>(B.size()) - 1 - i;

        // 获取 A 和 B 的维度值
        int dimA = (idxA >= 0) ? A[idxA] : 1;
        int dimB = (idxB >= 0) ? B[idxB] : 1;

        // 3. 检查是否可以广播
        if (dimA != dimB && dimA != 1 && dimB != 1) {
            IT_ASSERT(false);
        }

        // 4. 取最大值作为输出形状的维度
        output[maxDim - 1 - i] = std::max(dimA, dimB);
    }
    return output;
}

/**
 * @brief 将可能为负数的轴索引转换为正数索引
 *
 * 在张量操作中，轴索引可以是正数或负数：
 * - 正数：从前往后的索引，0表示第一维
 * - 负数：从后往前的索引，-1表示最后一维
 *
 * 此函数将负数索引转换为等价的正数索引
 *
 * 例如，对于一个 rank=3 的张量：
 * - axis=0 -> 0 (第一维)
 * - axis=1 -> 1 (第二维)
 * - axis=2 -> 2 (第三维)
 * - axis=-1 -> 2 (第三维)
 * - axis=-2 -> 1 (第二维)
 * - axis=-3 -> 0 (第一维)
 *
 * @param axis 输入的轴索引，可以是正数或负数
 * @param rank 张量的维度数
 * @return 转换后的正数轴索引
 */
int get_real_axis(const int &axis, const int &rank) {
    // 断言：rank必须大于等于1
    IT_ASSERT(rank >= 1);
    // 断言：axis必须在有效范围内 [-rank, rank-1]
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));

    int newAxis;
    if (axis < 0) {
        // 负数索引：从后往前，转换为正数索引
        newAxis = rank + axis;
    } else {
        // 正数索引：直接使用
        newAxis = axis;
    }
    return newAxis;
}

/**
 * @brief 将扁平化的一维索引转换为多维索引
 *
 * 此函数将张量的线性索引（flat index）转换为对应的多维坐标索引。
 * 用于将 Flatten 展平的索引还原为原始形状的多维索引。
 *
 * 工作原理：
 * - 从最后一维开始，用形状维度对索引取模得到当前维度的坐标
 * - 用整数除法得到剩余的索引值
 * - 重复直到计算完所有维度
 *
 * 例如，对于一个 shape=(2, 3, 4) 的张量：
 * - inputN=5 会转换为索引 [0, 1, 1]
 *   - 最后一个维度: 5 % 4 = 1, 5 / 4 = 1
 *   - 中间维度: 1 % 3 = 1, 1 / 3 = 0
 *   - 第一个维度: 0 % 2 = 0
 *
 * @param inputN 扁平化的一维索引
 * @param shape 张量的形状
 * @return 多维索引坐标
 */
Shape locate_index(size_t inputN, const Shape &shape) {
    // 初始化结果数组，大小与shape相同
    Shape ans(shape.size());
    // 使用反向迭代器从最后一维开始计算
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        // 使用std::div同时得到商和余数
        auto div = std::div(inputN, *j++);
        // 余数即为当前维度的索引
        *i++ = div.rem;
        // 商作为下一个维度的输入
        inputN = div.quot;
    }
    return ans;
}

/**
 * @brief 将多维索引转换为扁平化的一维索引（使用步幅stride）
 *
 * 此函数是 locate_index 的逆操作，将多维坐标索引转换为线性的内存偏移量。
 * 步幅（stride）表示在每个维度上移动一个索引所需要跨越的元素数量。
 *
 * 工作原理：
 * - 对每个维度的索引进行模运算，确保索引在有效范围内（支持广播）
 * - 将每个维度的索引乘以对应的步幅，累加得到线性偏移量
 *
 * 例如，对于一个 shape=(2, 3, 4) 的张量，stride=(12, 4, 1)：
 * - 索引 [1, 2, 3] 会转换为：
 *   - 1 * 12 + 2 * 4 + 3 * 1 = 12 + 8 + 3 = 23
 *
 * @param shapeIndex 多维坐标索引
 * @param shape 张量的形状
 * @param stride 每个维度的步幅
 * @return 扁平化的一维索引
 */
size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    // 断言：确保数组的维度一致
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());

    for (size_t i = 0; i < shape.size(); ++i) {
        // 对索引取模，处理广播情况（索引可能超出形状范围）
        index[i] = shapeIndex[i] % shape[i];
        // 累加每个维度的偏移量
        ans += index[i] * stride[i];
    }
    return ans;
}

/**
 * @brief 将设备枚举类型转换为字符串表示
 *
 * 此函数将 Device 枚举值转换为可读的字符串形式，
 * 用于日志输出、调试信息显示等场景。
 *
 * @param device 设备类型枚举值
 * @return 设备类型的字符串表示
 */
std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        // 遇到未实现的设备类型，终止程序
        IT_TODO_HALT();
    }
}

/**
 * @brief 将内核属性转换为字符串表示
 *
 * KernelAttrs 是一个元组，包含设备和操作类型等信息。
 * 此函数将这些属性转换为易读的字符串格式，便于日志输出和调试。
 *
 * @param kernelAttrs 内核属性元组，包含设备类型和操作类型
 * @return 格式化的内核属性字符串
 */
std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    // 从元组中提取设备类型并转换为字符串
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    // 从元组中提取操作类型并转换为字符串
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    // 拼接成格式化的字符串
    return deviceStr + ", " + opStr;
}

} // namespace infini
