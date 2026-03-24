# 基于 Voxel Hash 的 TSDF 融合与 Marching Cubes 完整示例

这个项目提供一个可直接编译运行的最小完整实现，包含：

- 稀疏 `voxel hash` 体素块管理
- 深度帧到 `TSDF` 的逐帧融合
- 激光散斑点云到 `TSDF` 的逐帧融合
- 基于 `Marching Cubes` 的等值面网格提取
- 合成“散斑式”带噪深度序列生成
- ASCII `PLY` 网格导出
- `PLY` 预览图生成脚本

项目定位是教学和工程起步版本：结构完整、依赖少、易扩展，不追求 GPU 实时性能。

## 目录结构

- `include/tsdfmc/math_types.h`: 基础向量、旋转、位姿、相机模型
- `include/tsdfmc/dataset_io.h`: `datasets` 深度图数据集读取
- `include/tsdfmc/marching_cubes.h`: Marching Cubes 查表与单体素格三角化
- `include/tsdfmc/voxel_hash_tsdf.h`: 稀疏哈希 TSDF 融合与网格提取
- `src/main.cpp`: 合成深度序列、融合流程、PLY 导出、CLI
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

数据集模式下：

- 直接读取 `depth.png` 中的 16-bit 深度图
- 用 `intrinsic.txt` 中的相机内参与深度缩放恢复米制深度
- 结合 `pose.txt` 或 `speckle_pose.txt` 将深度帧融合到世界坐标系下的 TSDF

### 3. Marching Cubes

- 把体素样本视作规则标量网格节点
- 对每个体素单元读取 8 个角点 TSDF
- 调用标准 `edgeTable / triTable` 生成三角片

## 当前版本的边界

- 只做 CPU 融合，没有做并行优化
- 深度采样使用最近邻，没有做双线性插值
- 网格导出为了简单起见，按三角形直接复制顶点，没有做顶点去重
- 数据集深度 PNG 解码当前走 Windows WIC 路径

## 后续扩展建议

- 接入真实深度图输入接口，例如 PNG / EXR / TUM / Replica
- 用姿态文件替代内置轨迹
- 增加法向估计和顶点去重
- 改成多线程 / SIMD / CUDA
- 加入体素颜色融合和彩色网格导出
