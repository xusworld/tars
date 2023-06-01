# 开发计划

## 2023/05

Implement basic concepts of a deep learning framework.

### Tars IR
tars ir
- graph
- op
- tensor
- types

### Offline Tools
1. Onnx IR --> tars IR

### Runtime Design
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


## 2023/06

Ready to run the first demo!

### Core

1. Shape Inference
2. Cuda Device/Runtime
3. Cpu(x86) Device/Runtime

### Ops

CPU(x86)

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

 CUDA

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
