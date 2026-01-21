#include "core/allocator.h"
#include <utility>

namespace infini {
// 构造函数：初始化分配器
Allocator::Allocator(Runtime runtime) : runtime(runtime) {
    used = 0;      // 当前正在使用的内存大小（字节）
    peak = 0;      // 历史上使用过的最大内存大小（峰值）
    ptr = nullptr; // 真正的内存块指针，初始为空（延迟分配）

    // 设置内存对齐要求为 8 字节（uint64_t 的大小）
    // 因为这是当前支持的最长数据类型
    alignment = sizeof(uint64_t);
}

// 析构函数：释放真正分配的内存
Allocator::~Allocator() {
    if (this->ptr != nullptr) {
        // 如果已经真正分配了内存，则释放它
        runtime->dealloc(this->ptr);
    }
}

// 分配内存：返回偏移量（相对于内存块起始位置的字节数）
size_t Allocator::alloc(size_t size) {
    // 断言：此时还没有真正分配内存（规划阶段）
    IT_ASSERT(this->ptr == nullptr);

    // 将请求的大小对齐到 alignment 的倍数
    // 例如：size=10 -> 对齐后=16（假设 alignment=8）
    size = this->getAlignedSize(size);

    // =================================== 作业
    // ===================================
    // TODO: 设计一个算法来分配内存，返回起始地址偏移量
    //
    // 需要考虑的问题：
    // 1. 返回什么偏移量？（提示：从哪里开始分配？）
    // 2. 如何更新 used？（提示：分配后，正在使用的内存增加了多少？）
    // 3. 如何更新 peak？（提示：peak 记录的是峰值，需要和谁比较？）
    // 4. 是否需要记录这块内存的信息？（考虑：如果后面 free 了，如何重用？）
    // =================================== 作业
    // ===================================
    for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
        size_t block_offset = it->first;
        size_t block_size = it->second;

        if (block_size >= size) {
            size_t offset = block_offset;
            used += size;
            peak = std::max(peak, used);

            freeBlocks.erase(it);

            size_t remaining_size = block_size - size;
            if (remaining_size > 0) {
                size_t remaining_offset = block_offset + size;
                freeBlocks[remaining_offset] = remaining_size;
            }
            return offset;
        }
    }

    size_t offset = used;
    used += size;
    peak = std::max(peak, used);

    return offset;
}

// 释放内存：回收指定偏移量和大小的内存块
void Allocator::free(size_t addr, size_t size) {
    // 断言：此时还没有真正分配内存（规划阶段）
    IT_ASSERT(this->ptr == nullptr);

    // 将大小对齐到 alignment 的倍数
    size = getAlignedSize(size);

    // =================================== 作业 ===================================
    // TODO: 设计一个算法来回收内存
    //
    // 需要考虑的问题：
    // 1. 如何更新 used？（提示：释放后，正在使用的内存减少了多少？）
    // 2. 需要记录这块被释放的内存吗？（考虑：下次 alloc 能否重用这块内存？）
    // 3. 如何避免内存碎片？（提示：可能需要某种数据结构来管理空闲块）
    // =================================== 作业 ===================================
    // 1. 更新 used：减少正在使用的内存
    used -= size;

    // 2. 记录这块被释放的内存到 freeBlocks
    freeBlocks[addr] = size;

    // 3. 合并相邻的空闲块以避免内存碎片

    // 3.1 检查是否可以与下一个空闲块合并
    auto next_it = freeBlocks.find(addr + size);
    if (next_it != freeBlocks.end()) {
        size += next_it -> second;
        freeBlocks.erase(next_it);
        freeBlocks[addr] = size;
    }

    // 3.2 检查是否可以与前一个空闲块合并
    for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
        if (it->first + it->second == addr) {
            size_t merged_size = it->second + size;
            size_t merged_addr = it->first;
            freeBlocks.erase(it);
            freeBlocks.erase(addr);
            freeBlocks[merged_addr] = merged_size;
            break;
        }
    }
}

// 获取真正的内存指针
void *Allocator::getPtr() {
    if (this->ptr == nullptr) {
        // 第一次调用时，根据 peak 值一次性分配所需的内存
        this->ptr = runtime->alloc(this->peak);
        printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
    }
    return this->ptr; // 返回内存块的起始地址
}

// 计算对齐后的大小
// 例如：size=10, alignment=8 -> 返回 16
//      size=16, alignment=8 -> 返回 16
//      size=17, alignment=8 -> 返回 24
size_t Allocator::getAlignedSize(size_t size) {
    // 公式：((size - 1) / alignment + 1) * alignment
    // 等价于：向上取整到 alignment 的倍数
    return ((size - 1) / this->alignment + 1) * this->alignment;
}

// 打印分配器的统计信息
void Allocator::info() {
    std::cout << "Used memory: " << this->used                 // 当前使用量
              << ", peak memory: " << this->peak << std::endl; // 峰值
}
} // namespace infini

