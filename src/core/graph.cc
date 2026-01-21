#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>


namespace infini {

// 添加算子并建立连接关系
// 参数: op - 要添加的算子对象
// 功能:
// 1. 将算子添加到图中，标记图需要重新排序
// 2. 建立输入tensor与该算子的连接
// 3. 建立输出tensor与该算子的连接
// 4. 维护算子的前驱后继关系
void GraphObj::addOperatorAndConnect(const Operator &op) {
    // 标记图需要重新拓扑排序
    sorted = false;
    // 将算子添加到算子列表中
    ops.push_back(op);

    // 处理输入tensor的连接关系
    for (auto &input : op->getInputs()) {
        if (input) {
            // 将当前算子添加到输入tensor的目标算子列表中
            input->addTarget(op);

            // 如果输入tensor有源算子，建立前驱后继关系
            if (auto pred = input->getSource()) {
                // 源算子将当前算子添加为后继
                pred->addSuccessors(op);
                // 当前算子将源算子添加为前驱
                op->addPredecessors(pred);
            }
        }
    }

    // 处理输出tensor的连接关系
    for (auto &output : op->getOutputs()) {
        if (output) {
            // 设置当前算子为输出tensor的源算子
            output->setSource(op);

            // 对于输出tensor的每个目标算子，建立前驱后继关系
            for (auto &succ : output->getTargets()) {
                // 目标算子将当前算子添加为前驱
                succ->addPredecessors(op);
                // 当前算子将目标算子添加为后继
                op->addSuccessors(succ);
            }
        }
    }
}

// 将图转换为字符串表示形式
// 返回值: 包含图结构信息的字符串
// 功能: 输出所有tensor和算子的详细信息，包括算子的前驱后继关系
string GraphObj::toString() const {
    std::ostringstream oss;

    // 输出所有tensor的信息
    oss << "Graph Tensors:\n";
    for (const auto &tensor : tensors)
        oss << tensor << "\n";

    // 输出所有算子的信息
    oss << "Graph operators:\n";
    for (const auto &op : ops) {
        // 收集算子的前驱和后继算子的GUID
        vector<UidBaseType> preds, succs;

        // 获取所有前驱算子的GUID
        for (auto &o : op->getPredecessors())
            preds.emplace_back(o->getGuid());

        // 获取所有后继算子的GUID
        for (auto &o : op->getSuccessors())
            succs.emplace_back(o->getGuid());

        // 输出算子的详细信息
        oss << "OP " << op->getGuid();
        oss << ", pred " << vecToString(preds); // 前驱算子列表
        oss << ", succ " << vecToString(succs); // 后继算子列表
        oss << ", " << op << "\n";              // 算子类型信息
    }
    return oss.str();
}

// 拓扑排序算法
// 返回值: 排序成功返回true，存在环导致无法排序返回false
// 功能: 对图中的算子进行拓扑排序，确保算子按依赖关系排列
// 算法思想:
// 1. 使用Kahn算法的变种
// 2. 找出没有前驱的算子优先处理
// 3. 处理后更新其后继算子的状态
// 4. 如果无法处理所有算子，说明存在环
bool GraphObj::topo_sort() {
    if (this->sorted) { // ① 检查是否已排序
        return true;
    }

    std::vector<Operator> sorted;            // ② 存储排序结果
    std::unordered_set<OperatorObj *> flags; // ③ 标记已处理的算子
    sorted.reserve(ops.size());
    flags.reserve(ops.size());

    while (sorted.size() < ops.size()) { // ④ 循环直到所有算子都被排序
        auto modified = false;           // ⑤ 标记本轮是否有进展

        for (auto const &op : ops) { // ⑥ 遍历所有算子
            if (auto const &inputs = op->getInputs();
                // ⑦ 算子未被处理过
                flags.find(op.get()) == flags.end() &&
                // ⑧ 所有输入的前驱算子都已被处理
                std::all_of(inputs.begin(), inputs.end(),
                            [&flags](auto const &input) {
                                auto ptr = input->getSource().get();
                                return !ptr || flags.find(ptr) != flags.end();
                            })) {

                modified = true;
                sorted.emplace_back(op); // ⑨ 添加到排序结果
                flags.insert(op.get());  // ⑩ 标记为已处理
            }
        }

        if (!modified) { // ⑪ 本轮无进展 = 存在环
            return false;
        }
    }

    this->ops = std::move(sorted); // ⑫ 更新算子列表
    return this->sorted = true;
}

void GraphObj::optimize() {
    // 辅助函数：检查两个permutation是否互为逆置换
    // p2[p1[i]] == i 对所有 i 成立
    auto isInversePermutation = [](const vector<int> &p1,
                                   const vector<int> &p2) -> bool {
        if (p1.size() != p2.size())
            return false;
        for (size_t i = 0; i < p1.size(); i++) {
            if (p2[p1[i]] != (int)i) {
                return false;
            }
        }
        return true;
    };

    // 辅助函数：检查permutation是否只交换最后两个维度
    // 例如：{0,1,3,2} 对于4维tensor是交换最后两维
    auto isLastTwoDimsSwap = [](const vector<int> &permute, int rank) -> bool {
        if (permute.size() != (size_t)rank)
            return false;
        // 前面的维度保持不变
        for (int i = 0; i < rank - 2; i++) {
            if (permute[i] != i) {
                return false;
            }
        }
        // 最后两个维度互换
        return permute[rank - 2] == rank - 1 && permute[rank - 1] == rank - 2;
    };

    // 规则1：删除冗余的Transpose算子对
    // 当两个连续的Transpose互为逆置换时，它们可以被消除
    auto removeRedundantTransposePairs = [&]() -> bool {
        bool changed = false;

        // 使用while循环，因为删除算子会改变ops
        size_t i = 0;
        while (i < ops.size()) {
            Operator op1 = ops[i];

            // 检查是否是Transpose算子
            if (op1->getOpType() != OpType::Transpose) {
                i++;
                continue;
            }

            auto transpose1 = as<TransposeObj>(op1);
            Tensor middleTensor = transpose1->getOutput();

            // 检查输出tensor是否只有一个目标
            auto targets = middleTensor->getTargets();
            if (targets.size() != 1) {
                i++;
                continue;
            }

            // 检查目标是否也是Transpose
            Operator op2 = targets[0];
            if (op2->getOpType() != OpType::Transpose) {
                i++;
                continue;
            }

            auto transpose2 = as<TransposeObj>(op2);

            // 检查是否是逆置换
            if (!isInversePermutation(transpose1->getPermute(),
                                      transpose2->getPermute())) {
                i++;
                continue;
            }

            // 执行优化：删除这两个transpose，直接连接前后tensor
            Tensor inputTensor = transpose1->getInputs(0);
            Tensor outputTensor = transpose2->getOutput();

            // 获取inputTensor的源算子（可能为nullptr，表示是图输入）
            Operator inputSource = inputTensor->getSource();

            // 先复制targets列表，避免在遍历时修改
            auto successors = outputTensor->getTargets();

            // 让所有使用outputTensor的后继算子改用inputTensor
            for (auto succ : successors) {
                succ->replaceInput(outputTensor, inputTensor);
                inputTensor->addTarget(succ);
                // 清理后继中指向op2的前驱引用
                succ->removePredecessors(op2);

                // 如果inputTensor有源算子，建立新的前驱/后继关系
                if (inputSource) {
                    inputSource->addSuccessors(succ);
                    succ->addPredecessors(inputSource);
                }
            }

            // 清理op1的前驱和后继关系
            if (inputSource) {
                inputSource->removeSuccessors(op1);
            }
            op1->removeSuccessors(op2);

            // 清理op2的前驱
            op2->removePredecessors(op1);

            // 清理tensor连接
            inputTensor->removeTarget(op1);

            // 删除算子和tensor
            removeOperator(op1);
            removeOperator(op2);
            removeTensor(middleTensor);
            removeTensor(outputTensor);

            changed = true;
            // 不增加i，因为删除后当前位置是新的算子
        }

        return changed;
    };

    // 规则2：将Transpose融入Matmul的transA/transB属性
    // 当Matmul的输入来自一个仅交换最后两维的Transpose时，可以融合
    auto fuseTransposeIntoMatmul = [&]() -> bool {
        bool changed = false;

        for (size_t i = 0; i < ops.size(); i++) {
            Operator op = ops[i];

            // 检查是否是Matmul算子
            if (op->getOpType() != OpType::MatMul) {
                continue;
            }

            auto matmul = as<MatmulObj>(op);

            // 处理输入A
            Tensor inputA = matmul->getInputs(0);
            if (inputA->getSource() &&
                inputA->getSource()->getOpType() == OpType::Transpose &&
                inputA->getTargets().size() == 1) {

                auto trans = as<TransposeObj>(inputA->getSource());
                if (isLastTwoDimsSwap(trans->getPermute(), inputA->getRank())) {
                    Tensor originalA = trans->getInputs(0);
                    Operator transposeOp = inputA->getSource();

                    // 获取transposeOp的前驱（originalA的源算子）
                    Operator originalSource = originalA->getSource();

                    // 融合Transpose到transA（翻转原有值）
                    matmul->setTransA(!matmul->getTransA());
                    matmul->replaceInput(inputA, originalA);

                    // 更新tensor连接
                    originalA->removeTarget(transposeOp);
                    originalA->addTarget(matmul);

                    // 更新前驱/后继关系
                    matmul->removePredecessors(transposeOp);
                    if (originalSource) {
                        originalSource->removeSuccessors(transposeOp);
                        originalSource->addSuccessors(matmul);
                        matmul->addPredecessors(originalSource);
                    }

                    // 删除Transpose
                    removeOperator(transposeOp);
                    removeTensor(inputA);
                    changed = true;
                }
            }

            // 处理输入B（逻辑相同）
            // 注意：需要重新获取inputB，因为inputs可能已经改变
            Tensor inputB = matmul->getInputs(1);
            if (inputB->getSource() &&
                inputB->getSource()->getOpType() == OpType::Transpose &&
                inputB->getTargets().size() == 1) {

                auto trans = as<TransposeObj>(inputB->getSource());
                if (isLastTwoDimsSwap(trans->getPermute(), inputB->getRank())) {
                    Tensor originalB = trans->getInputs(0);
                    Operator transposeOp = inputB->getSource();

                    // 获取transposeOp的前驱（originalB的源算子）
                    Operator originalSource = originalB->getSource();

                    // 融合Transpose到transB（翻转原有值）
                    matmul->setTransB(!matmul->getTransB());
                    matmul->replaceInput(inputB, originalB);

                    // 更新tensor连接
                    originalB->removeTarget(transposeOp);
                    originalB->addTarget(matmul);

                    // 更新前驱/后继关系
                    matmul->removePredecessors(transposeOp);
                    if (originalSource) {
                        originalSource->removeSuccessors(transposeOp);
                        originalSource->addSuccessors(matmul);
                        matmul->addPredecessors(originalSource);
                    }

                    // 删除Transpose
                    removeOperator(transposeOp);
                    removeTensor(inputB);
                    changed = true;
                }
            }
        }

        return changed;
    };

    // 迭代执行优化，直到没有新的优化发生
    bool changed = true;
    int iteration = 0;
    const int MAX_ITERATIONS = 10; // 防止无限循环

    while (changed && iteration < MAX_ITERATIONS) {
        changed = false;
        iteration++;

        // 按顺序执行两个优化规则
        changed |= removeRedundantTransposePairs();
        changed |= fuseTransposeIntoMatmul();
    }
}

// 根据tensor的唯一标识符查找tensor
// 参数: fuid - tensor的唯一标识符
// 返回值: 找到的tensor对象，未找到返回nullptr
Tensor GraphObj::getTensor(int fuid) const {
    // 遍历图中所有tensor
    for (auto tensor : tensors) {
        // 匹配唯一标识符
        if (tensor->getFuid() == fuid) {
            return tensor; // 返回找到的tensor
        }
    }
    return nullptr; // 未找到匹配的tensor
}

// 形状推断：根据算子的输入形状推断输出形状
// 功能:
// 1. 遍历所有算子，调用各自的形状推断函数
// 2. 更新输出tensor的形状信息
// 3. 支持动态形状变化
// 注意: 需要图已经完成拓扑排序
void GraphObj::shape_infer() {
    for (auto &op : ops) {           // ① 按拓扑排序顺序遍历
        auto ans = op->inferShape(); // ② 调用算子的推断函数
        IT_ASSERT(ans.has_value());  // ③ 确保推断成功

        auto oldOutputs = op->getOutputs();                 // ④ 获取输出张量
        IT_ASSERT(ans.value().size() == oldOutputs.size()); // ⑤ 数量匹配

        for (int i = 0; i < (int)ans.value().size(); ++i) {
            auto newShape = ans.value()[i];           // ⑥ 推断的新形状
            auto oldShape = oldOutputs[i]->getDims(); // ⑦ 原始形状
            auto fuid = oldOutputs[i]->getFuid();     // ⑧ 唯一标识

            if (newShape != oldShape) {              // ⑨ 形状是否变化
                auto tensor = this->getTensor(fuid); // ⑩ 找到张量
                tensor->setShape(newShape);          // ⑪ 更新形状
            }
        }
    }
}

void GraphObj::dataMalloc() {
    IT_ASSERT(topo_sort() == true);

    // =================================== 作业 ===================================
    // TODO：利用 allocator 给计算图分配内存
    //
    // 背景知识：
    //   - 每个tensor需要占用一定的内存空间
    //   - tensor的大小 = shape中所有维度的乘积 × 数据类型的字节数
    //   - allocator是一个内存分配器，负责统一管理内存
    //
    // 实现思路：
    //   1. 遍历所有tensor
    //   2. 计算每个tensor需要的内存大小
    //      - 可以使用 tensor->size() 获取元素个数
    //      - 可以使用 tensor->getBytes() 获取字节数
    //   3. 调用 allocator.alloc(size) 分配内存
    //      - 返回一个内存指针
    //   4. 使用 tensor->setDataBlob(memPtr) 将内存绑定到tensor
    //
    // 进阶思考（可选）：
    //   - 如何复用内存？（生命周期不重叠的tensor可以共享内存）
    //   - 如何减少内存碎片？
    //   - 如何处理不同设备的内存（CPU vs GPU）？
    //
    // HINT:
    //   - 获取分配好的内存指针后，调用 tensor->setDataBlob() 绑定内存
    //   - allocator可能提供了 alloc(size) 方法来分配指定大小的内存
    // =================================== 作业 ===================================

    // 实现说明：
    // Allocator采用两阶段设计：规划阶段(ptr==nullptr)和执行阶段(ptr!=nullptr)
    // 必须先完成所有alloc()调用，再调用getPtr()获取真实内存

    // 如果没有tensor，直接返回
    if (tensors.empty()) {
        return;
    }

    // 阶段1：规划 - 收集所有tensor的内存偏移量
    std::vector<size_t> offsets;
    for (auto &tensor : tensors) {
        size_t size = tensor->getBytes();
        size_t offset = allocator.alloc(size);
        offsets.push_back(offset);
    }

    // 阶段2：分配 - 获取真实内存基址
    void *basePtr = allocator.getPtr();

    // 阶段3：绑定 - 为每个tensor设置内存
    for (size_t i = 0; i < tensors.size(); ++i) {
        void *memPtr = static_cast<char *>(basePtr) + offsets[i];
        Blob blob = make_ref<BlobObj>(runtime, memPtr);
        tensors[i]->setDataBlob(blob);
    }

    allocator.info();
}

// 创建并添加新的tensor到图中
// 参数:
//   dim - tensor的形状维度
//   dtype - 数据类型
// 返回值: 新创建的tensor对象
Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
    // 创建新的tensor对象并添加到tensor列表中
    return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

// 将已存在的tensor添加到图中
// 参数: tensor - 要添加的tensor对象
// 返回值: 添加的tensor对象
// 功能: 检查运行时兼容性后将tensor添加到图中
Tensor GraphObj::addTensor(const Tensor &tensor) {
    // 检查tensor的运行时环境与当前图是否兼容
    IT_ASSERT(tensor->getRuntime() == runtime,
              std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                  tensor->getRuntime()->toString() + " to " +
                  runtime->toString());
    tensors.emplace_back(tensor); // 添加到tensor列表
    return tensor;
}

// 批量添加tensor到图中
// 参数: tensors - 要添加的tensor向量
// 返回值: 添加的tensor向量
// 功能: 遍历tensor向量，逐个调用addTensor添加
TensorVec GraphObj::addTensor(const TensorVec &tensors) {
    for (auto &t : tensors)
        addTensor(t); // 逐个添加tensor
    return tensors;
}

// 检查图的有效性
// 返回值: 图有效返回true，否则抛出断言错误
// 功能:
// 1. 检查tensor的连接关系是否正确
// 2. 检查算子的输入输出tensor是否都在图中
// 3. 检查算子的前驱后继关系是否正确
// 4. 检查tensor的唯一标识符是否重复
bool GraphObj::checkValid() const {
    // 检查所有tensor的连接关系
    for (auto tensor : tensors) {
        // tensor不能既没有目标算子又没有源算子（孤立的tensor）
        IT_ASSERT(
            !(tensor->getTargets().size() == 0 && nullptr == tensor->getSource()));

        // 检查tensor的所有目标算子是否都在图的算子列表中
        for (auto op : tensor->getTargets()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
        }

        // 检查tensor的源算子是否在图的算子列表中
        auto op = tensor->getSource();
        IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
    }

    // 检查所有算子的输入输出tensor和前驱后继关系
    for (auto op : ops) {
        // 检查输入tensor是否都在图的tensor列表中
        for (auto tensor : op->getInputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }

        // 检查输出tensor是否都在图的tensor列表中
        for (auto tensor : op->getOutputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }

        // 检查前驱算子是否都在图的算子列表中
        for (auto pre : op->getPredecessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
        }

        // 检查后继算子是否都在图的算子列表中
        for (auto suc : op->getSuccessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
        }
    }

    // 检查tensor的唯一标识符是否重复
    std::set<UidBaseType> s; // 用于检查重复的集合
    for (auto tensor : tensors) {
        int cnt = s.count(tensor->getFuid());                   // 检查是否已存在
        IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid())); // 不允许重复
        s.insert(tensor->getFuid());                            // 添加到集合
    }

    return true; // 所有检查通过，图有效
}

} // namespace infini
