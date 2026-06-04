# BlockGCN 代码 Pipeline 与核心算法讲解

> 配套论文：**BlockGCN: Redefine Topology Awareness for Skeleton-Based Action Recognition** (CVPR 2024)
> 论文 PDF：[`BlockGCN_CVPR24/paper.pdf`](BlockGCN_CVPR24/paper.pdf)
> 核心实现：[`model/BlockGCN.py`](model/BlockGCN.py)

---

## 1. 论文要解决的两个问题

| 问题 | 论文 | 代码对应 |
| --- | --- | --- |
| **P1** 可学习邻接矩阵在训练中"灾难性遗忘"骨骼物理拓扑 | Sec.3.1 / Fig.3 | `unit_gcn.rpe` + `unit_gcn.hops`（静态）+ `Topo` / `TopoTrans`（动态） |
| **P2** 多关系建模 (multi-relational) 用 GC ensemble 太重，权重矩阵冗余 | Sec.3.1 / Tab.6 | `unit_gcn.fc2 = Conv2d(..., groups=num_heads)` 实现块对角权重 |

对应的两大创新：

1. **Topological Encoding**（双路并行）
   - **Static**：用骨骼图上的最短路径距离 SPD 查表，得到 `B_ij = e_{d_ij}`（论文 Eq.2）。**只有 embedding 表 E 在训练，SPD 矩阵 hops 不会被更新，骨骼连通性永远不会被"遗忘"。**
   - **Dynamic**：对每个动作序列用 Vietoris-Rips Complex + Persistent Homology 算 barcode，再用可微 vectorization Ψ⁰ 投影到 GCN 隐藏维度（论文 Eq.4）。这是"动作级"的拓扑指纹。

2. **BlockGC**：把权重矩阵 W 设计成块对角形式（论文 Eq.6）。参数量从 O(d²) 降到 O(d²/K)，且让 K 个特征组独立建模 K 种不同的语义关系。

最终 spatial 聚合公式（论文 Eq.5）：

```
H^(l) = σ( (A^(l) + B^(l)) · (H^(l-1) + C^(l)) · W^(l) )
        └────┬────┘    └────┬────┘   └─┬─┘
             │              │           └─ BlockGC 块对角权重 (论文 Eq.6 的 W_k)
             │              └─ 主特征 + dynamic topo encoding C
             └─ 可学习邻接 A + static topo encoding B (查 SPD 表)
```

---

## 2. 目录结构

```
.
├── main.py                   # 入口：解析 config → 训练/测试主循环
├── model/BlockGCN.py         # 核心模型（已加详细注释）
├── feeders/
│   ├── feeder_ntu.py         # NTU RGB+D / NTU 120 dataset
│   ├── feeder_ucla.py        # NW-UCLA dataset
│   ├── tools.py              # 数据增强 (random_rot, valid_crop_resize 等)
│   └── bone_pairs.py         # bone 模态用的关节配对表
├── graph/
│   ├── ntu_rgb_d.py          # NTU 25 关节骨骼图定义 (inward/outward 边)
│   ├── ucla.py               # UCLA 20 关节骨骼图
│   └── tools.py              # 邻接矩阵构造工具 (edge2mat, normalize_digraph)
├── config/                   # 各数据集 / 各模态的 yaml 配置
├── train.sh                  # 训练脚本 (4 模态分别训练)
├── evaluate.sh               # 评估脚本
├── ensemble.py / ensemble.sh # 多模态结果 score-level fusion
└── BlockGCN_CVPR24/paper.md  # 论文 markdown 版
```

---

## 3. 整体调用链 (Pipeline)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  shell:  python main.py --config config/.../default.yaml                 │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  main.py: __main__                                                        │
│    1. get_parser() → 命令行参数                                            │
│    2. yaml.load(config) 覆盖默认值                                         │
│    3. init_seed(seed)                                                     │
│    4. processor = Processor(arg)                                          │
│    5. processor.start()                                                   │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Processor.__init__                                                       │
│    · save_arg()  → 把配置写入 work_dir/config.yaml                         │
│    · load_model()                                                         │
│        Model = import_class('model.BlockGCN.Model')                       │
│        self.model = Model(**model_args).cuda()                            │
│        self.loss  = CrossEntropyLoss()                                    │
│    · load_optimizer()  → SGD + Nesterov + weight_decay                    │
│    · load_data()                                                          │
│        Feeder = import_class('feeders.feeder_ntu.Feeder')                 │
│        DataLoader(train/test)                                             │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Processor.start()  (train phase)                                         │
│    for epoch in range(num_epoch):                                         │
│        ├─ train(epoch)   一个 epoch 训练                                   │
│        └─ eval(epoch)    在 test set 上验证                                │
│    最后加载 best epoch 权重，再在 test set 上保存预测 score                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.1 单个训练 step (Processor.train)

```
for batch_idx, (joint, data, label, index) in enumerate(loader):
    # joint : 原始关节坐标 (N, 3, T, V, M) —— 永远是 joint 模态，用于计算 Persistent Homology
    # data  : 当前模态的输入 (N, 3, T, V, M) —— 可能是 joint / bone / joint_motion / bone_motion
    # label : 类别 id

    with autocast(enabled=True):
        output, _ = self.model(data, F.one_hot(label, num_classes), joint)
        loss      = self.loss(output, label)        # 标准 cross-entropy

    optimizer.zero_grad()
    scaler.scale(loss).backward()                    # AMP 反传
    scaler.step(optimizer)
    scaler.update()
```

**注意点**：
- 训练使用 **AMP 混合精度** (`torch.cuda.amp`)，但代码注释说该模型 AMP 反而更慢 —— 是个保留遗迹。
- `joint` 和 `data` 是两份独立张量，因为 Persistent Homology 必须用真实欧氏距离 (论文 `w_ij = ||x_i - x_j||_2`)，所以即使你跑 bone 或 motion 模态，PH 分支也始终用原始 joint 坐标。
- 多模态融合在外部做：`train.sh` 用同一份代码分别训练 4 个模态，再用 `ensemble.py` 做 score-level fusion。

---

## 4. 数据流：`Feeder.__getitem__`

对应文件：[`feeders/feeder_ntu.py`](feeders/feeder_ntu.py)

```
.npz 文件 (x_train: (N, T, M*V*C))
        │
        ▼ reshape → (N, C=3, T, V=25, M=2)   # NTU 25 关节，最多 2 人
        │
        ▼ valid_crop_resize  → 把有效帧数随机裁剪到 window_size=64
        │
        ▼ random_rot         → ±0.3 rad 随机旋转 (训练时)
        │
        ├──► joint = 原始坐标 (无论后面要算哪个模态都保留)
        │
        ▼ 根据 self.bone / self.vel 计算最终输入:
        │   · joint   模态: 每帧坐标减去 spine center (v=20)，保留 spine 自身轨迹
        │   · bone    模态: bone[v1] = joint[v1] - joint[v2]，按 ntu_pairs 配对
        │   · motion  模态: data[:-1] = data[1:] - data[:-1] (相邻帧差)
        │
        ▼ 返回 (joint, data, label, index)
```

---

## 5. 模型 forward 详解：`model/BlockGCN.py`

> 一个完整 forward 的 tensor shape 演变（NTU60，batch=N，T=64，V=25，M=2）：

```
joint, data: (N, 3, 64, 25, 2)
                │
   ┌────────────┴────────────────────────────────┐
   │                                             │
   ▼ Topo 分支                                    ▼ 主分支
   joint → mean(M)     (N, 3, 64, 25)            data
       → pair diff     (N, 3, 64, 25, 25)        rearrange → (N·M·T, V, C)
       → mean(T)       (N, 3, 25, 25)            to_joint_embedding (3 → 128)
       → L2_norm       (N, 25, 25)               + pos_embedding (per joint)
       → min-max norm                            data_bn
       → VR Complex → Barcodes                   reshape → (N·M, 128, 64, 25)
       → StructureElementLayer → (N, 64)              │
                       │                              │
                       ▼ TopoTrans(t_i)               │
                       (N·M, 128 or 256, 1, 1)        │
                       │                              │
                       └──── broadcast add ──────────►│
                                                       ▼
                              ┌────────────────────────────┐
                              │ l1 ~ l10 (TCN_GCN_unit)    │
                              │   unit_gcn  + MultiScale   │
                              │   (BlockGC + Static Topo)  TCN
                              └──────────────┬─────────────┘
                                             ▼
                              (N·M, 256, 16, 25)  ← stride=2 两次时间下采样
                                             ▼
                       view(N, M, 256, T'·V) → mean → (N, 256)
                                             ▼
                       fc(256 → num_class)  → logits
```

### 5.1 `unit_gcn.forward`（论文核心：BlockGC + Static Topo Encoding）

```python
pos_emb = self.rpe[:, :, self.hops]   # (3, H, V, V) ← 查 SPD 表得到 B_ij

for i in range(3):                     # 3 个并行 GC 分支 (self/inward/outward)
    w1 = self.fc1[i] / L2(self.fc1[i])             # 可学习邻接 A_k 归一化
    w1 = w1 + pos_emb[i] / L2(pos_emb[i])           # A + B  ← 论文 Eq.5

    # 把 channel 分成 num_heads (=K) 组
    x_in = x.view(N, H, C//H, T, V)
    z    = einsum("nhctv, hvw -> nhctw", x_in, w1)  # 每组独立 spatial 聚合

    z    = self.fc2[i](z)                           # groups=H 的 Conv2d → 块对角 W
    y    = y + z if y is not None else z

y = bn(y) + down(x); y = relu(y)
```

关键设计：
- `self.fc1` 是可学习邻接，但 `pos_emb` (来自 SPD) 始终 **加** 进去 —— 即使 fc1 被遗忘，骨骼拓扑也通过 rpe 表保留。
- `self.fc2 = Conv2d(..., groups=num_heads)` 是 BlockGC 的关键：让 Conv1×1 沿通道维分组，等价于 W 在结构上是块对角矩阵。

### 5.2 `Topo.forward` (Dynamic Encoding)

```python
x = x.mean(1)                            # 沿 person 维平均
x = x.unsqueeze(-1) - x.unsqueeze(-2)    # 关节对差向量 → 距离矩阵
x = x.mean(-3); x = L2_norm(x)           # 沿 T 平均 + 取模 → V×V 距离矩阵 w_ij
x = (x - min) / (max - min)              # 归一化
x = self.vr(x)                           # VietorisRipsComplex → persistence diagram
x = make_tensor(x)
x = self.pl(x)                           # StructureElementLayer → 64 维向量 (Ψ⁰)
```

### 5.3 每层注入 dynamic encoding

```python
x = self.l1(x + self.t0(a))   # a 是 (N, 64) 的 barcode 向量化结果
x = self.l2(x + self.t1(a))   # t_i = TopoTrans = MLP(64 → C_l) + BN + ReLU → (N·M, C_l, 1, 1)
...
```

`t_i(a)` 沿 T、V 维广播加到主特征上，相当于给每帧每个关节都附上同一份"动作级"拓扑指纹。

---

## 6. 损失与训练

- **Loss**：仅 `CrossEntropyLoss`（论文 Sec.4 强调："**we employ the standard cross-entropy loss in all our experiments to ensure an impartial assessment of our architecture**"）
- **Optimizer**：SGD + Nesterov，momentum 0.9，weight_decay 4e-4（NTU），2e-4（UCLA）
- **LR schedule**：base_lr=0.05 → 在 epoch 110 / 120 各 ×0.1；可选 warm_up_epoch
- **Batch size**：NTU 64，UCLA 16；**window_size=64 帧**
- **Epoch**：默认 110；保存最佳 epoch 权重
- **EMA**：可选 (`--ema true`)，默认关闭

---

## 7. 多模态推理 (4-Stream Fusion)

论文采用 **J + B + JM + BM** 四模态 score-level 融合（Tab.2）。代码对应：

```
train.sh                        evaluate.sh                ensemble.py
   ├─ joint    .yaml               joint    weights           ┐
   ├─ bone     .yaml      →        bone     weights     →    weighted sum of softmax
   ├─ vel      .yaml               vel      weights           ┘
   └─ bone_vel .yaml               bone_vel weights
```

只需切 yaml 里的 `bone: True/False` 和 `vel: True/False`，模型代码不变；Feeder 内部根据这两个 flag 算不同的输入张量。

---

## 8. 一句话总结

> BlockGCN 用 **预计算的 SPD 表 + 持续同调 barcode** 两条独立通路注入"不可被遗忘"的骨骼拓扑（解决 P1），用 **groups Conv 实现的块对角权重 W** 让 K 组特征独立建模 K 种语义关系（解决 P2）。10 层 (BlockGC + MultiScale TCN) 堆叠 + GAP + Linear → 4 模态 ensemble 出 SOTA 精度（NTU120 X-Sub 90.3%，参数 1.3M，比 FR-Head 少 35%）。
