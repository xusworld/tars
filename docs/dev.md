# 项目介绍

## 项目结构

传统推理框架的架构设计。

## 项目特性



# 开发计划

2023年5月上旬

## 离线工具（部分完成）
1. Onnx IR --> tars IR

## 设计 tars IR
tars ir
- graph
- op
- tensor
- types

2023年5月中旬
## 运行时模块
1. core
    - Allocator
    - Buffer
    - Tensor
    - Memory Pool
    - Thread Pool  
2. device
    - Runtime Interface
    - Device Interface
3. tools
    - String Tools


## 基础算子开发

2023年5月下旬

### CPU(x86)

1. Elementwise
2. Binary
3. Reduce
4. Pool
5. Reshape
6. Slice
7. Flatten
8. Permute
9. ArgMax
10. BatchNorm
11. Softmax
12. InnerProduct
13. Normalize 
14. Resize
15. Gemm
16. Scale
17. Conv
18. LRN

### CUDA

1. Elementwise
2. Binary
3. Reduce
4. Pool
5. Reshape
6. Slice
7. Flatten
8. Permute
9. ArgMax
10. BatchNorm
11. Softmax
12. InnerProduct
13. Normalize 
14. Resize
15. Gemm
16. Scale
17. Conv
18. LRN
