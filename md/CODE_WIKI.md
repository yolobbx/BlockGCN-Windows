# BlockGCN Code Wiki

## 项目概述

**BlockGCN** 是一个基于骨架的动作识别深度学习模型，出自 CVPR 2024 论文 [BlockGCN: Redefining Topology Awareness for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.pdf)。

该模型通过重新定义拓扑感知机制，在骨架动作识别任务中实现了更高的准确率，同时保持了更少的模型参数。

---

## 项目架构

```
BlockGCN/
├── config/                    # 配置文件目录
│   ├── nturgbd-cross-subject/ # NTU 60 Cross-Subject 配置
│   ├── nturgbd-cross-view/    # NTU 60 Cross-View 配置
│   ├── nturgbd120-cross-set/  # NTU 120 Cross-Set 配置
│   ├── nturgbd120-cross-subject/ # NTU 120 Cross-Subject 配置
│   └── ucla/                  # UCLA 数据集配置
├── data/                      # 数据处理脚本
│   ├── ntu/                   # NTU 60 数据预处理
│   ├── ntu120/                # NTU 120 数据预处理
│   └── NW-UCLA/               # UCLA 数据集
├── feeders/                   # 数据加载器
│   ├── feeder_ntu.py          # NTU 数据集加载器
│   ├── feeder_ucla.py         # UCLA 数据集加载器
│   ├── bone_pairs.py          # 骨骼对定义
│   └── tools.py               # 数据处理工具函数
├── graph/                     # 图结构定义
│   ├── ntu_rgb_d.py           # NTU 骨架图结构
│   ├── ucla.py                # UCLA 骨架图结构
│   └── tools.py               # 图构建工具函数
├── model/                     # 模型定义
│   └── BlockGCN.py            # 核心模型实现
├── torchlight/                # 训练工具库
│   └── torchlight/
│       ├── gpu.py             # GPU 相关工具
│       └── util.py            # 通用工具函数
├── main.py                    # 主训练/测试入口
├── ensemble.py                # 多模态集成脚本
├── train.sh                   # 训练启动脚本
├── evaluate.sh                # 测试启动脚本
└── ensemble.sh                # 集成启动脚本
```

---

## 主要模块详解

### 1. 核心模型 (`model/BlockGCN.py`)

#### 1.1 Model 类（主模型）

**职责**：整合所有模块，完成骨架动作识别的端到端处理

**关键参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `num_class` | int | 动作类别数 |
| `num_point` | int | 关节数量（NTU: 25, UCLA: 20） |
| `num_person` | int | 每帧人数 |
| `graph` | str | 图结构类名 |
| `in_channels` | int | 输入特征维度（默认3：x,y,z坐标） |
| `drop_out` | float | Dropout 比率 |
| `adaptive` | bool | 是否使用自适应邻接矩阵 |
| `alpha` | bool | 是否启用拓扑编码 |

**网络结构**：
```
输入 (N, C, T, V, M)
    ↓
to_joint_embedding (3 → 128)
    ↓
10层 TCN_GCN_unit (l1-l10)
    ↓
全局池化 + Dropout
    ↓
全连接层 → 类别输出
```

#### 1.2 TCN_GCN_unit 类

**职责**：时空卷积单元，结合时序卷积(TCN)和图卷积(GCN)

**结构**：
- `gcn1`: unit_gcn 图卷积层
- `tcn1`: MultiScale_TemporalConv 多尺度时序卷积
- 残差连接

#### 1.3 unit_gcn 类（图卷积单元）

**职责**：实现基于块对角结构的图卷积操作

**核心特性**：
- **多头注意力机制**：将特征维度分成多个头（默认为8）
- **k-hop距离编码**：使用相对位置编码(RPE)编码节点间的距离
- **L2归一化**：对邻接矩阵进行归一化处理

**关键参数**：
| 参数 | 说明 |
|------|------|
| `in_channels` | 输入通道数 |
| `out_channels` | 输出通道数 |
| `A` | 邻接矩阵 |
| `adaptive` | 是否自适应学习邻接矩阵 |
| `alpha` | 是否使用统计拓扑编码 |

#### 1.4 MultiScale_TemporalConv 类（多尺度时序卷积）

**职责**：并行使用多个不同膨胀率的时序卷积分支，捕捉不同范围的时序依赖

**分支结构**：
- 4个膨胀卷积分支（dilations: [1, 2, 3, 4]）
- 1个最大池化分支
- 1个标准卷积分支

#### 1.5 TopoTrans 类（拓扑变换）

**职责**：将拓扑信息转换为可学习的嵌入向量

#### 1.6 Topo 类（拓扑提取）

**职责**：使用 Vietoris-Rips 复形提取骨架拓扑结构

**依赖**：`torch_topological` 库

---

### 2. 数据加载器 (`feeders/`)

#### 2.1 Feeder_ntu（NTU 数据加载器）

**职责**：加载和预处理 NTU RGB+D 数据集

**数据格式**：
- 输入: NPZ 文件
- 形状: `(N, C, T, V, M)`
  - N: 样本数
  - C: 通道数 (3: x, y, z坐标)
  - T: 时间帧数
  - V: 关节数 (25)
  - M: 人数 (最多2人)

**关键方法**：
| 方法 | 说明 |
|------|------|
| `load_data()` | 加载 NPZ 数据文件 |
| `__getitem__()` | 返回单个样本，支持数据增强 |
| `top_k()` | 计算 Top-K 准确率 |

**数据增强**：
- `random_rot`: 随机旋转骨架
- `bone`: 骨骼模态（相邻关节向量差）
- `vel`: 速度模态（一阶差分）

#### 2.2 Feeder_ucla（UCLA 数据加载器）

**职责**：加载和预处理 NW-UCLA 数据集

**数据格式**：
- 形状: `(C, T, V, 1)` (UCLA 只有1人)

**关键差异**：
- 关节数: 20（比 NTU 少5个）
- 人数: 固定为1

---

### 3. 图结构定义 (`graph/`)

#### 3.1 Graph_ntu_rgb_d（NTU 骨架图）

**职责**：定义 NTU 数据集的骨架拓扑结构

**关键属性**：
| 属性 | 形状 | 说明 |
|------|------|------|
| `A` | (3, 25, 25) | 邻接矩阵（自环、向心、离心） |
| `A1` | (11, 11) | 粗粒度邻接矩阵（11关节） |
| `A2` | (5, 5) | 极粗粒度邻接矩阵（5关节） |
| `A_binary` | (25, 25) | 二值邻接矩阵 |
| `A_norm` | (25, 25) | 归一化邻接矩阵 |

**NTU 25关节定义**：
```
0: pelvis (骨盆)
1-3:  spine (脊柱)
4-6:  neck/head (颈部/头部)
7-9:  left shoulder/arm (左肩/臂)
10-12: right shoulder/arm (右肩/臂)
13-15: left hand (左手)
16-18: right hand (右手)
19-21: left/right hip (髋部)
22-25: legs (腿部)
```

#### 3.2 Graph_ucla（UCLA 骨架图）

**职责**：定义 UCLA 数据集的骨架拓扑结构

**关键差异**：
- 关节数: 20（无脚部关节）

---

### 4. 训练框架 (`main.py`)

#### 4.1 Processor 类

**职责**：管理整个训练/测试流程

**核心方法**：
| 方法 | 说明 |
|------|------|
| `load_model()` | 加载模型并初始化权重 |
| `load_optimizer()` | 配置优化器（SGD/NAdam/Adam/AdamW） |
| `load_data()` | 创建数据加载器 |
| `train()` | 执行单个 epoch 的训练 |
| `eval()` | 执行模型评估 |
| `start()` | 启动训练或测试流程 |

**训练特性**：
- **混合精度训练**：使用 `torch.cuda.amp` 加速
- **指数移动平均 (EMA)**：可选的模型参数平滑
- **学习率调度**：支持预热（warm-up）和阶梯衰减
- **多GPU并行**：使用 `DataParallel` 支持多卡训练
- **TensorBoard日志**：记录训练过程

#### 4.2 关键工具函数

| 函数 | 说明 |
|------|------|
| `import_class()` | 动态导入类 |
| `init_seed()` | 初始化随机种子保证可复现性 |
| `ema_update()` | EMA 模型参数更新 |
| `get_parser()` | 命令行参数解析 |

---

### 5. 多模态集成 (`ensemble.py`)

**职责**：融合多个模态的预测结果

**支持的模态**：
- `joint`: 关节坐标
- `bone`: 骨骼向量
- `motion`: 运动速度

**融合策略**：加权求和
```python
final_score = joint * α₁ + bone * α₂ + motion * α₃
```

**默认权重**：
- NTU: `[0.6, 0.7, 0.35, 0.2]`
- UCLA: `[0.7, 0.5, 0.6, 0.2]`

---

## 关键类与函数速查

### 模型组件

| 类名 | 文件 | 功能 |
|------|------|------|
| `Model` | `model/BlockGCN.py` | 主模型类 |
| `TCN_GCN_unit` | `model/BlockGCN.py` | 时空卷积单元 |
| `unit_gcn` | `model/BlockGCN.py` | 图卷积单元 |
| `MultiScale_TemporalConv` | `model/BlockGCN.py` | 多尺度时序卷积 |
| `TemporalConv` | `model/BlockGCN.py` | 基础时序卷积 |
| `unit_tcn` | `model/BlockGCN.py` | 时间卷积块 |
| `TopoTrans` | `model/BlockGCN.py` | 拓扑变换 |
| `Topo` | `model/BlockGCN.py` | 拓扑提取 |

### 数据处理

| 类名/函数 | 文件 | 功能 |
|-----------|------|------|
| `Feeder` | `feeders/feeder_ntu.py` | NTU 数据加载器 |
| `Feeder` | `feeders/feeder_ucla.py` | UCLA 数据加载器 |
| `valid_crop_resize()` | `feeders/tools.py` | 裁剪并resize序列 |
| `random_choose()` | `feeders/tools.py` | 随机选择序列片段 |
| `random_move()` | `feeders/tools.py` | 随机仿射变换 |
| `random_rot()` | `feeders/tools.py` | 随机旋转 |

### 图工具

| 函数 | 文件 | 功能 |
|------|------|------|
| `get_spatial_graph()` | `graph/tools.py` | 构建空间邻接图 |
| `normalize_adjacency_matrix()` | `graph/tools.py` | 归一化邻接矩阵 |
| `k_adjacency()` | `graph/tools.py` | k阶邻接矩阵 |
| `Graph` | `graph/ntu_rgb_d.py` | NTU图结构 |
| `Graph` | `graph/ucla.py` | UCLA图结构 |

---

## 依赖关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                 │
│                     (训练/测试入口)                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │Processor │    │ Feeder   │    │ Model    │
    │ (训练器) │    │(数据加载) │    │(网络结构) │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │              │               │
         │              │        ┌──────┴──────┐
         │              │        ▼             ▼
         │              │   ┌────────┐   ┌────────┐
         │              │   │Graph   │   │Topo    │
         │              │   │(图结构) │   │(拓扑)   │
         │              │   └────────┘   └────────┘
         │              │
    ┌────┴─────┐   ┌────┴─────┐
    │Optimizer │   │ torch    │
    │torchlight│   │ DataLoader│
    └──────────┘   └──────────┘
```

---

## 项目运行方式

### 环境准备

```bash
# 1. 安装 torchlight
pip install -e torchlight

# 2. 安装核心依赖
pip install torch numpy pyyaml scikit-learn tqdm einops
pip install torch-topological
pip install tensorboardX
```

### 数据准备

```bash
# NTU RGB+D 60
# 1. 下载数据集: https://rose1.ntu.edu.sg/dataset/actionRecognition
# 2. 提取到 ./data/nturgbd_raw/

# 数据预处理
cd ./data/ntu
python get_raw_skes_data.py      # 提取骨架数据
python get_raw_denoised_data.py  # 去噪处理
python seq_transformation.py     # 变换到第一帧中心
```

### 训练

```bash
# NTU 60 Cross-Subject
python main.py \
    --config config/nturgbd-cross-subject/default.yaml \
    --model model.BlockGCN.Model \
    --work-dir work_dir/ntu60/csub/joint \
    --device 0 1

# UCLA
python main.py \
    --config config/ucla/default.yaml \
    --model model.BlockGCN.Model \
    --work-dir work_dir/ucla/joint \
    --device 0
```

### 测试

```bash
python main.py \
    --weights work_dir/xxx/runs-xx-xx.pt \
    --phase test \
    --config config/xxx.yaml \
    --model model.BlockGCN.Model \
    --work-dir work_dir/test
```

### 多模态集成

```bash
# 集成 joint + bone + motion 四个模态
python ensemble.py \
    --dataset NW-UCLA \
    --joint-dir work_dir/ucla/joint \
    --bone-dir work_dir/ucla/bone \
    --joint-motion-dir work_dir/ucla/vel \
    --bone-motion-dir work_dir/ucla/bone_vel
```

---

## 配置说明

### 配置文件结构 (YAML)

```yaml
# 基本设置
work_dir: ./work_dir/output

# 数据加载器配置
feeder: feeders.feeder_ntu.Feeder
train_feeder_args:
  data_path: data/ntu60/NTU60_CS.npz
  split: train
  window_size: 64
  random_rot: True
  bone: False
  vel: False

test_feeder_args:
  data_path: data/ntu60/NTU60_CS.npz
  split: test
  window_size: 64

# 模型配置
model: model.BlockGCN.Model
model_args:
  num_class: 60
  num_point: 25
  num_person: 2
  graph: graph.ntu_rgb_d.Graph
  graph_args:
    labeling_mode: 'spatial'

# 优化器配置
weight_decay: 0.0004
base_lr: 0.05
lr_decay_rate: 0.1
step: [110, 120]
warm_up_epoch: 5
batch_size: 64
num_epoch: 140

# 设备配置
device: [0, 1, 2, 3]
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_lr` | 0.05 | 基础学习率 |
| `batch_size` | 64 | 批量大小 |
| `num_epoch` | 140 | 训练轮数 |
| `window_size` | 64 | 输入序列长度 |
| `weight_decay` | 0.0004 | 权重衰减 |
| `lr_decay_rate` | 0.1 | 学习率衰减率 |

---

## 数据流图

```
原始视频/深度图
      │
      ▼
骨骼提取 (NTU120: 25关节 × 3坐标 × T帧 × M人)
      │
      ▼
┌─────────────────────────────────────┐
│           Feeder (数据加载器)         │
├─────────────────────────────────────┤
│ 1. 加载 NPZ 数据                      │
│ 2. 数据增强 (旋转、平移、缩放)          │
│ 3. 骨骼/速度模态计算                   │
│ 4. 序列裁剪和 resize                   │
└─────────────────────────────────────┘
      │
      ▼
Model Input: (N, C=3, T=64, V=25, M=2)
      │
      ▼
┌─────────────────────────────────────┐
│     BlockGCN Model (10层)            │
├─────────────────────────────────────┤
│ 1. to_joint_embedding (3→128)       │
│ 2. 10 × TCN_GCN_unit:               │
│    - unit_gcn (图卷积 + k-hop编码)   │
│    - MultiScale_TemporalConv         │
│ 3. TopoTrans (拓扑信息注入)          │
│ 4. 全局平均池化                       │
│ 5. Dropout + FC → 类别概率          │
└─────────────────────────────────────┘
      │
      ▼
分类损失计算 + 准确率评估
```

---

## 预训练模型

建议的模型存储位置：
```
BlockGCN_pretrained_weights/
├── ntu60/
│   ├── csub/
│   │   ├── joint/
│   │   ├── bone/
│   │   ├── vel/
│   │   └── bone_vel/
│   └── cview/
│       └── ...
├── ntu120/
│   ├── csub/
│   └── cset/
└── ucla/
```

---

## 常见问题

### 1. 显存不足
- 减小 `batch_size`
- 减少 `num_worker` 数量
- 使用单GPU训练

### 2. 数据加载慢
- 设置 `use_mmap=True`
- 增加 `num_worker`
- 启用 `prefetch_factor`

### 3. 训练不收敛
- 检查学习率设置
- 确认数据格式正确
- 验证 `num_class` 与数据集匹配

---

## 扩展指南

### 添加新数据集

1. 在 `feeders/` 创建新的 feeder 类
2. 在 `graph/` 定义对应的图结构
3. 在 `config/` 创建配置文件
4. 修改 `ensemble.py` 支持新数据集

### 修改模型结构

1. 修改 `model/BlockGCN.py` 中的 `Model` 类
2. 调整 `TCN_GCN_unit` 中的层数或类型
3. 更新配置文件中的 `model_args`

---

## 参考资料

- [BlockGCN 论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.pdf)
- [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)
- [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN)
- [NTU RGB+D 数据集](https://rose1.ntu.edu.sg/dataset/actionRecognition)
- [torch-topological](https://github.com/aidos-ai/pytorch-topological)
