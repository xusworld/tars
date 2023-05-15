
Device 模块的设计从上到下包含以下几个层级

1. Device: 设备级别的顶层抽象 
2. Runtime: 负责内存管理、线程管理、监测数据
3. 内存管理和线程管理
- MemoryPool: 负责内存管理
- ThreadPool: 负责线程管理
4. Allocator: CpuAllocator 负责申请和释放 Host Memory; CudaAllocator 负责申请和释放 Cuda Memory
5. Buffer: 内存抽象


开发计划

1. 设计 Buffer，并编写测试示例
2. 设计 Allocator，并编写测试示例
3. 设计内存池，并编写测试示例
4. 设计线程池，并编写测试示例
5. 设计 Runtime，并编写测试示例
6. 设计 Device，并编写测试示例

截止 2023/05/14 的目标是实现Binary, Elementwise, Reduce 接口，并成功执行。