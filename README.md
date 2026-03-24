# 基于 Voxel Hash 的 TSDF 融合与 Marching Cubes 完整示例

这个项目提供一个可直接编译运行的最小完整实现，包含：

- 稀疏 `voxel hash` 体素块管理
- 深度帧到 `TSDF` 的逐帧融合
- 可选的 CUDA 深度帧积分加速
- 激光散斑点云到 `TSDF` 的逐帧融合
- 基于 `Marching Cubes` 的等值面网格提取
- 合成“散斑式”带噪深度序列生成
- ASCII `PLY` 网格导出
- `PLY` 预览图生成脚本

项目定位是教学和工程起步版本：结构完整、依赖少、易扩展。当前版本保留了轻量 CPU 实现，同时参考 `VoxelHashing-master` 增加了一个可选 CUDA 融合后端。

## 目录结构

- `include/tsdfmc/math_types.h`: 基础向量、旋转、位姿、相机模型
- `include/tsdfmc/dataset_io.h`: `datasets` 深度图数据集读取
- `include/tsdfmc/reconstruction_backend.h`: CPU / GPU 融合后端选择
- `include/tsdfmc/marching_cubes.h`: Marching Cubes 查表与单体素格三角化
- `include/tsdfmc/voxel_hash_tsdf.h`: 稀疏哈希 TSDF 融合与网格提取
- `src/main.cpp`: 合成深度序列、融合流程、PLY 导出、CLI
- `src/reconstruction_backend.cpp`: CPU / GPU 后端分发
- `src/cuda_block_allocator.cu`: CUDA 活跃块哈希分配与紧凑化
- `src/cuda_tsdf_integrator.cu`: CUDA 活跃体素块积分
- `src/cuda_mesh_extractor.cu`: CUDA Marching Cubes 提取
- `tools/visualize_ply.py`: `.ply` 顶点预览和 PNG 导出

## 依赖

- `CMake >= 3.16`
- `g++ >= 9` 或等价的 C++17 编译器
- `liboctomap-dev`

说明：

- 这里没有链接 `OctoMap` 库，只是直接复用其头文件 `octomap/MCTables.h` 里的标准 `Marching Cubes` 查表。
- 若系统缺少该头文件，构建阶段会报错并提示安装 `liboctomap-dev`。

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake liboctomap-dev
```

## 构建

```bash
cmake -S . -B build
cmake --build build -j
```

说明：

- 如果 CMake 能找到 CUDA Toolkit，会自动启用 CUDA 后端
- 如果找不到 CUDA Toolkit，会自动回退成纯 CPU 构建
- 当前仓库里的 GPU 适配借鉴了 `VoxelHashing-master` 的“活跃块紧凑化后并行处理”思路，已经覆盖了活跃块生成、深度帧积分和 Marching Cubes 提取
- CUDA 路径内部已经开始使用可复用的持久工作区，避免每帧重复 `cudaMalloc/cudaFree`，并让 active block 生成与积分共用一次深度上传
- 但还没有完整搬运它的设备端哈希分配、垃圾回收、raycasting 和 streaming 管线

Windows + CUDA 说明：

- 当前环境下，`Visual Studio 2026 + nvcc 13.1` 更稳妥的方式是从 `VsDevCmd.bat` 里用 `NMake Makefiles` 生成
- 仓库里的 CMake 已经自动为这组版本加上 `--allow-unsupported-compiler`
- 示例：

```powershell
cmd /c "\"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat\" -arch=x64 && cmake -G \"NMake Makefiles\" -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=D:/cudatoolkit/bin/nvcc.exe && cmake --build build-cuda -- /nologo"
```

## 运行

```bash
./build/tsdf_voxel_hash_demo
```

默认会：

1. 使用硬编码到程序中的 `datasets` 目录
2. 逐帧读取 `depth.png + pose.txt`
3. 将深度帧融合进稀疏 TSDF
4. 用 Marching Cubes 提取网格
5. 输出到 `output/reconstruction.ply`

可以显式指定融合后端：

```bash
./build/tsdf_voxel_hash_demo --backend auto
./build/tsdf_voxel_hash_demo --backend cpu
./build/tsdf_voxel_hash_demo --backend gpu
```

说明：

- `auto`：优先使用 CUDA，若当前构建或运行环境不可用则自动回退 CPU
- `cpu`：强制使用 CPU
- `gpu`：强制使用 CUDA；如果当前构建没有 CUDA 支持会直接报错

如果想运行旧的合成深度示例：

```bash
./build/tsdf_voxel_hash_demo \
  --synthetic \
  --output output/demo_mesh.ply \
  --frames 36 \
  --width 160 \
  --height 120 \
  --voxel-size 0.015 \
  --truncation 0.05
```

## 读取 `datasets` 深度图数据集

这个仓库已经支持直接读取仓库根目录下 `datasets` 中的文件：

- `datasets/intrinsic.txt`
- `datasets/<frame_id>/pose.txt`
- `datasets/<frame_id>/speckle_pose.txt`
- `datasets/<frame_id>/depth.png`

注意：

- `depth.png` 是标准 16-bit 深度 PNG
- 深度值会按 `intrinsic.txt` 里的 `depth_scale` 转成米
- 默认数据集路径会在编译时硬编码成当前仓库的 `datasets`
- 程序会打印逐帧加载和融合日志。

示例：

```bash
./build/tsdf_voxel_hash_demo \
  --frame-start 0 \
  --frame-end 200 \
  --frame-step 20 \
  --pose-file pose.txt \
  --output output/datasets_0_200_step20.ply
```

如果想改用 `speckle_pose.txt`：

```bash
./build/tsdf_voxel_hash_demo \
  --frame-step 10 \
  --pose-file speckle_pose.txt \
  --output output/datasets_speckle_pose.ply
```

推荐先从较大的 `frame-step` 开始，例如 `10` 或 `20`，确认结果后再减小步长。

## 可视化 `.ply`

提供了一个独立脚本，用 `matplotlib` 把 `.ply` 顶点采样后渲染成 PNG 预览图，也可以加 `--show` 弹出交互窗口。

渲染本项目输出的 `.ply`：

```bash
python3 tools/visualize_ply.py \
  output/hualaohu_0_200_step20.ply \
  --output output/hualaohu_0_200_step20.png
```

渲染数据集自带的大型二进制 `PLY`：

```bash
python3 tools/visualize_ply.py \
  hualaohu2shanchanzi-20260210/Fused-0.2mm.ply \
  --max-points 5000 \
  --output output/fused_preview.png
```

脚本会打印：

- 文件格式
- 顶点数量
- 面数量
- 采样后的可视化点数
- 输出 PNG 路径

## 实现说明

### 1. Voxel Hash

- 以 `8x8x8` 为一个稀疏体素块
- 用 `std::unordered_map<BlockKey, Block>` 存活跃块
- 根据深度点及其截断带前后位置分配邻域块

### 2. TSDF 融合

- 对已激活块中的每个体素中心投影到当前深度图
- 用 `depth - z` 计算 signed distance
- 对 `[-truncation, +truncation]` 范围做截断归一化
- 用累计权重做平均融合
- 当前 CPU / GPU 路径都会先把当前帧附近的活跃块收集出来，再只更新这些块；GPU 路径会先在设备端把 block key 写入临时哈希表并 compact 成唯一列表，再回到主机端落地到当前 `unordered_map`

数据集模式下：

- 直接读取 `depth.png` 中的 16-bit 深度图
- 用 `intrinsic.txt` 中的相机内参与深度缩放恢复米制深度
- 结合 `pose.txt` 或 `speckle_pose.txt` 将深度帧融合到世界坐标系下的 TSDF

### 3. Marching Cubes

- 把体素样本视作规则标量网格节点
- 对每个体素单元读取 8 个角点 TSDF
- 调用标准 `edgeTable / triTable` 生成三角片
- 当前 GPU 路径会先把有观测的 block 紧凑化，再在 CUDA 上做两阶段 triangle count + triangle emit

## 当前版本的边界

- 深度采样使用最近邻，没有做双线性插值
- 网格导出为了简单起见，按三角形直接复制顶点，没有做顶点去重
- 数据集深度 PNG 解码当前走 Windows WIC 路径
- GPU 路径当前已经覆盖活跃块生成、TSDF 积分和 Marching Cubes，但最终的稀疏哈希表存储仍保留在主机端 `unordered_map`
- GPU 路径虽然开始有持久工作区，但体素块本体还不是设备端常驻结构，当前每帧积分后仍会同步回主机端
- GPU 路径依赖本机安装 CUDA Toolkit；本仓库不会自动下载 CUDA

## 后续扩展建议

- 接入真实深度图输入接口，例如 PNG / EXR / TUM / Replica
- 用姿态文件替代内置轨迹
- 增加法向估计和顶点去重
- 改成多线程 / SIMD / CUDA
- 加入体素颜色融合和彩色网格导出
