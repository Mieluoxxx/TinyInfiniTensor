#include "operators/unary.h"

namespace infini {
// ==================== UnaryObj 类实现 ====================
// 一元操作符的基类，所有单输入单输出的操作符都可以继承此类

/**
 * @brief UnaryObj 构造函数
 * @param type 操作符类型（如 Relu、Sigmoid 等）
 * @param graph 计算图对象指针
 * @param input 输入张量
 * @param output 输出张量
 */
UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(type, {input}, {output}) {
    // 验证操作符在计算图中的有效 性
    IT_ASSERT(checkValid(graph));
}

/**
 * @brief 推断一元操作符的输出形状
 * @param inputs 输入张量向量（只有一个输入）
 * @return 输出张量的形状向量
 *
 * 一元操作符不改变张量的形状，输出形状与输入形状相同
 */
optional<vector<Shape>> UnaryObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

/**
 * @brief 将一元操作符转换为字符串表示
 * @return 操作符的字符串描述
 *
 * 格式：操作类型[GUID](形状,input=输入GUID,output=输出GUID)
 */
std::string UnaryObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

// ==================== ClipObj 类实现 ====================
// Clip 操作符：将张量值裁剪到指定范围内 [min, max]

/**
 * @brief ClipObj 构造函数
 * @param graph 计算图对象指针
 * @param input 输入张量
 * @param output 输出张量
 * @param min 可选的最小值，如果为 nullopt 则不限制下界
 * @param max 可选的最大值，如果为 nullopt 则不限制上界
 *
 * Clip 操作将输入张量的每个元素限制在 [min, max] 范围内
 * 小于 min 的值会被设置为 min，大于 max 的值会被设置为 max
 */
ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor output,
                 std::optional<float> min, std::optional<float> max)
    : OperatorObj(OpType::Clip, {input}, {output}), minValue(min),
      maxValue(max) {
    IT_ASSERT(checkValid(graph));
}

/**
 * @brief 推断 Clip 操作符的输出形状
 * @param inputs 输入张量向量（只有一个输入）
 * @return 输出张量的形状向量
 *
 * Clip 操作不改变张量的形状，输出形状与输入形状相同
 *
 * @note 作业：需要实现此函数
 * @see https://onnx.ai/onnx/operators/onnx__Clip.html#clip-13
 */
optional<vector<Shape>> ClipObj::inferShape(const TensorVec &inputs) {
    // =================================== 作业 ===================================
    // TODO：返回经过 clip 操作后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Clip.html#clip-13
    // =================================== 作业 ===================================
    return {{inputs[0]->getDims()}};
}

/**
 * @brief 将 Clip 操作符转换为字符串表示
 * @return 操作符的字符串描述
 *
 * 格式：Clip[GUID](形状,input=输入GUID,output=输出GUID)
 */
std::string ClipObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

// ==================== CastObj 类实现 ====================
// Cast 操作符：将张量的数据类型从一种类型转换为另一种类型

/**
 * @brief CastObj 构造函数
 * @param graph 计算图对象指针
 * @param input 输入张量
 * @param output 输出张量
 * @param type 类型转换类型（如 Float2Int32、Int322Float 等）
 *
 * Cast 操作将输入张量的数据类型转换为指定的目标类型
 * 支持多种类型转换，包括浮点数、整数之间的相互转换
 */
CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
    : OperatorObj(OpType::Cast, {input}, {output}), castType(type) {
    IT_ASSERT(checkValid(graph));
}

/**
 * @brief 推断 Cast 操作符的输出数据类型
 * @param inputs 输入张量向量（只有一个输入）
 * @return 输出张量的数据类型向量
 *
 * Cast 操作会改变张量的数据类型，但不改变形状
 * 应返回包含目标数据类型的向量
 *
 * @note 作业：需要实现此函数
 * @see https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
 */
vector<DataType> CastObj::inferDataType(const TensorVec &inputs) const {
    // =================================== 作业 ===================================
    // TODO：返回经过 cast 操作后, 输出 tensor 的数目和数据类型
    // REF_FILE: src/core/operator.cc
    // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
    // =================================== 作业 ===================================
    return {getOutputDataType()};
}

/**
 * @brief 推断 Cast 操作符的输出形状
 * @param inputs 输入张量向量（只有一个输入）
 * @return 输出张量的形状向量
 *
 * Cast 操作不改变张量的形状，输出形状与输入形状相同
 *
 * @note 作业：需要实现此函数
 * @see https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
 */
optional<vector<Shape>> CastObj::inferShape(const TensorVec &inputs) {
    // =================================== 作业 ===================================
    // TODO：返回经过 cast 操作后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
    // =================================== 作业 ===================================
    return {{inputs[0]->getDims()}};
}

/**
 * @brief 将 Cast 操作符转换为字符串表示
 * @return 操作符的字符串描述
 *
 * 格式：Cast[GUID](output=输出GUID)
 */
std::string CastObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

/**
 * @brief 获取 Cast 操作的目标数据类型
 * @return 目标数据类型
 *
 * 根据 castType 枚举值返回对应的目标数据类型
 * 支持 24 种不同的类型转换组合
 */
DataType CastObj::getOutputDataType() const {
    switch (castType) {
    case CastType::Float2Float16:
        return DataType::Float16;
    case CastType::Float2Int64:
        return DataType::Int64;
    case CastType::Float2Int32:
        return DataType::Int32;
    case CastType::Float2Int16:
        return DataType::Int16;
    case CastType::Float2Int8:
        return DataType::Int8;
    case CastType::Int322Float:
        return DataType::Float32;
    case CastType::Int322Int8:
        return DataType::Int8;
    case CastType::Int322Int16:
        return DataType::Int16;
    case CastType::Int162Float:
        return DataType::Float32;
    case CastType::Int162Int32:
        return DataType::Int32;
    case CastType::Int82Float:
        return DataType::Float32;
    case CastType::Int82Int16:
        return DataType::Int16;
    case CastType::Int82Int32:
        return DataType::Int32;
    case CastType::Uint82Float:
        return DataType::Float32;
    case CastType::Uint82Int32:
        return DataType::Int32;
    case CastType::Uint82Int64:
        return DataType::Int64;
    case CastType::Int322Int64:
        return DataType::Int64;
    case CastType::Int642Int32:
        return DataType::Int32;
    case CastType::Int642Uint32:
        return DataType::UInt32;
    case CastType::Int642Float:
        return DataType::Float32;
    case CastType::Uint322Int64:
        return DataType::Int64;
    case CastType::Float162Float:
        return DataType::Float32;
    case CastType::BFloat162Float:
        return DataType::Float32;
    case CastType::Float2BFloat16:
        return DataType::BFloat16;
    case CastType::Float2Float:
        return DataType::Float32;
    default:
        IT_TODO_HALT();
    }
}
}; // namespace infini
