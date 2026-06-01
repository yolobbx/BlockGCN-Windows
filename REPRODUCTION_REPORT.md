# BlockGCN 复现报告（PPT 大纲版）

> 复现仓库：`BlockGCN-Windows`（基于 [BlockGCN-main 官方版](https://github.com/ZhouYuxuanYX/BlockGCN) 适配 Windows + 单 GPU）
> 论文：**BlockGCN: Redefine Topology Awareness for Skeleton-Based Action Recognition** (CVPR 2024)
> 论文位置：[`BlockGCN_CVPR24/paper.pdf`](BlockGCN_CVPR24/paper.pdf) / [`paper.md`](BlockGCN_CVPR24/paper.md)

文档结构（建议 PPT 一节对应一章）：

1. 环境准备 & 依赖问题
2. 代码改动（按报错类型分类）
3. 配置文件与运行脚本改动
4. 核心算法理解（论文 ↔ 代码对照）
5. 实验结果（占位，等截图）

---

## 一、环境准备 & 依赖问题

### 1.1 目标环境

- OS: **Windows**（官方版仅在 Linux + Tesla V100 上验证过）
- GPU: **单卡 NVIDIA**
- Python: 3.8+
- 框架: PyTorch + tensorboardX + einops + torch_topological

### 1.2 数据集

| 数据集 | 关节数 | 人数 | 类别 | 划分 |
|---|---|---|---|---|
| NTU RGB+D 60 | 25 | ≤2 | 60 | X-Sub / X-View |
| NTU RGB+D 120 | 25 | ≤2 | 120 | X-Sub / X-Set |
| NW-UCLA | 20 | 1 | 10 | View 1+2 train, View 3 test |

数据预处理产出 `.npz`（含 `x_train/y_train/x_test/y_test`）放在 `data/` 下，由 `feeders/feeder_ntu.py` 或 `feeders/feeder_ucla.py` 加载。

---

## 二、代码改动（按报错分类）

### 2.1 `main.py` —— 改动最集中的文件

#### 改动 ①：Windows 不支持 `resource` 模块

| 报错 | 原因 | 修复 |
|---|---|---|
| `ImportError: No module named 'resource'` 启动即崩 | Linux/Mac 系统模块，Windows 没有 | `try/except ImportError` 包住，并在使用处 `if resource is not None:` 守卫 |

```python
try:
    import resource
except ImportError:
    resource = None
...
if resource is not None:
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
```

#### 改动 ②：DataLoader 多进程在 Windows 上不稳

| 报错 | 原因 | 修复 |
|---|---|---|
| `BrokenPipeError` / `EOFError` / 反复重启子进程 | Windows 没有 `fork`，只能用 `spawn`；多进程 worker 需要主模块 import 保护 | `num_worker` 默认值 `16 → 0`（先单进程跑通） |
| `ValueError: prefetch_factor option could only be specified in multiprocessing` | `num_workers=0` 时 PyTorch 禁止再传 `prefetch_factor` | 动态拼装 `loader_args`：仅当 `num_worker > 0` 才加 `prefetch_factor` |

```python
loader_args = dict(num_workers=self.arg.num_worker, worker_init_fn=init_seed)
if self.arg.num_worker > 0:
    loader_args['prefetch_factor'] = 2
self.data_loader['train'] = torch.utils.data.DataLoader(
    dataset=Feeder(**self.arg.train_feeder_args),
    batch_size=self.arg.batch_size, shuffle=True, drop_last=True, **loader_args)
```

> 备注：跑通后建议把 `num_worker` 调成 2~4 加速 I/O；同时确保主入口有 `if __name__ == '__main__':` 守卫（Windows spawn 必备）。

#### 改动 ③：单卡 vs `DataParallel`

| 报错 | 原因 | 修复 |
|---|---|---|
| `AttributeError: 'Model' object has no attribute 'module'` | 原版始终用 `nn.DataParallel` 包装，处处写 `self.model.module.num_class`；单卡不包装就崩 | 增加 `unwrap_model()` 辅助函数，并在调用处改成 `self.model.num_class` |

```python
def unwrap_model(self):
    return self.model.module if hasattr(self.model, 'module') else self.model
```

> 仍可优化：把所有 `self.model.num_class` / `self.model.module.num_class` 统一替换为 `self.unwrap_model().num_class`，单卡/多卡都不用改。

#### 改动 ④：张量 device 不一致

| 报错 | 原因 | 修复 |
|---|---|---|
| `RuntimeError: Expected all tensors to be on the same device` | Feeder 返回的 `joint` / `data` 在 CPU，模型在 GPU；尤其 `joint` 这一路是 BlockGCN 新增分支，原版没有显式搬运 | 在 `train()` / `eval()` 主循环开头加 `data = data.to(self.output_device)`、`joint = joint.to(self.output_device)` |

#### 改动 ⑤：PyYAML 新版 API

| 报错 | 原因 | 修复 |
|---|---|---|
| `YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated` 或直接 raise | PyYAML 5.1+ 强制传 `Loader` | `yaml.load(f, Loader=yaml.FullLoader)` |

#### 改动 ⑥：警告噪声清理

```python
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
```
纯日志清爽化。

---

### 2.2 `model/BlockGCN.py` —— 模型层面的两处关键修复

#### 改动 ①：`TopoTrans` 写死了双人，UCLA 单人跑不通

| 报错 | 原因 | 修复 |
|---|---|---|
| 跑 UCLA 时 batch 维度对不上 / 训练 loss 不收敛 | 原版 `TopoTrans` 里硬编码 `x = x.repeat(2, 1)`（NTU 两人）；UCLA 是单人，注释里要求改成 `x = x` —— 等价于"切换数据集要改源码" | 引入 `num_person` 参数，通过 yaml `model_args.num_person` 联动 |

```python
class TopoTrans(nn.Module):
    def __init__(self, out_dim, num_person=2):
        ...
        self.num_person = num_person
    def forward(self, x):
        x = x.repeat(self.num_person, 1)
        ...
```

并在 `Model.__init__` 把 `num_person` 透传给 `t0..t9` 全部 10 层。

#### 改动 ②：`Topo` 模块输出在 CPU，模型在 GPU

| 报错 | 原因 | 修复 |
|---|---|---|
| `Expected all tensors to be on the same device` —— 出现在 `Topo.forward` 的 `make_tensor` 之后 | `torch_topological` 的 `make_tensor` 默认在 CPU 上生成张量 | 显式跟随输入 device |

```python
def forward(self, x):
    device = x.device
    ...
    x = make_tensor(x).to(device)   # ← 修复
    x = self.pl(x)
    return x
```

> 仍可优化：`t0..t9` 这 10 行重复代码可以收敛成 `nn.ModuleList`。

---

### 2.3 `torchlight/torchlight/util.py` —— PaviLogger 容错

| 报错 | 原因 | 修复 |
|---|---|---|
| `ImportError: No module named 'torchpack.runner.hooks'` | `torchpack` 是商汤内部依赖，pip 安装容易失败 | `try/except ImportError` + `PaviLogger = None` |

```python
try:
    from torchpack.runner.hooks import PaviLogger
except ImportError:
    PaviLogger = None
```

> 仍可优化：如果代码里真的有 `PaviLogger(...)` 调用点，需要再加 `if PaviLogger is None: skip`，否则会在调用时报 `'NoneType' object is not callable`。

### 2.4 `torchlight/__init__.py` —— 顶层导出

新增顶层 `__init__.py`：
```python
from .torchlight.util import IO, str2bool, str2dict, DictAction, import_class
from .torchlight.gpu import visible_gpu, occupy_gpu, ngpu
```

| 痛点 | 修复 |
|---|---|
| 原版 import 路径过深：`from torchlight.torchlight.util import ...` | 收敛成 `from torchlight import IO` |

---

## 三、配置与脚本改动

### 3.1 `train.sh` / `evaluate.sh`

| 改动 | 原因 |
|---|---|
| `--device 0 1` → `--device 0` | 本地单卡，使用第 2 张卡会 `CUDA error: invalid device ordinal` |
| 追加 UCLA 训练/测试命令 | 原版只放了 NTU 的命令 |
| 默认未注释的命令改成"当前要跑的那条" | 多条命令同时未注释会顺序执行 |

### 3.2 `config/ucla/default.yaml` —— 新增

原版根本没有这个 yaml，UCLA 需要自己写：

```yaml
model_args:
  num_class: 10
  num_point: 20
  num_person: 1       # ← 关键：触发 TopoTrans 的 num_person=1
  graph: graph.ucla.Graph
  ...
```

> ⚠️ 注意：yaml 里 `model: model.ctrgcn.Model` 是从其他 yaml 复制过来的，与 `train.sh` 里 `--model model.BlockGCN.Model` 不一致。命令行参数会覆盖 yaml 所以能跑通，但 yaml 自身有歧义，建议改成 `model.BlockGCN.Model`。

### 3.3 `requirements.txt` —— 新增

绝大多数行是注释，实际生效的是 `tensorboard / tqdm / scipy / torch` 等几行。

> 仍可优化：跑通后 `pip freeze > requirements.txt`，删掉所有注释行。

---

## 四、核心算法理解（论文 ↔ 代码）

### 4.1 论文要解决的两个问题

| 编号 | 问题 | 论文位置 |
|---|---|---|
| **P1** | 可学习邻接矩阵 A 在训练中"灾难性遗忘"骨骼物理拓扑（Fig.3 可视化：训练后 A 各层完全不同，远离骨骼连通） | Sec.3.1, Fig.3 |
| **P2** | Multi-relational 建模用 GC ensemble 太重，权重矩阵冗余 | Sec.3.1, Tab.1, Tab.6 |

### 4.2 两大创新

#### 4.2.1 Topological Encoding（双路并行注入拓扑）

**(a) Static Topological Encoding** —— 解决 P1，论文 Sec.3.2.1 / Eq.(2)

$$B_{ij} = e_{d_{i,j}}, \quad d_{i,j} = \min_{P \in \text{Paths}(\mathcal{G}_S)} \{|P|, P_1=v_i, P_{|P|}=v_j\}$$

- 把骨骼图上两节点的**最短路径距离 (SPD)** 作为索引去查一个可学习 embedding 表
- 只有 embedding 表 E 在训练时被更新，**SPD 矩阵本身永远不变** → 骨骼拓扑不会被遗忘

代码：`model/BlockGCN.py: unit_gcn`
```python
# 预计算 SPD（一次性，靠矩阵幂 + 差分得到"恰好 k 步可达"的 mask）
h1 = A.sum(0); h1[h1 != 0] = 1                  # 二值化邻接
h[0] = I; h[1] = h1
for i in range(2, V): h[i] = h[i-1] @ h1.T; h[i][h[i]!=0] = 1
for i in range(V-1, 0, -1):
    self.hops += i * (h[i] - h[i-1])             # hops[u,v] = SPD(u,v)

# 可学习的距离 → embedding 查表 (3 个分支 × num_heads × max_hop+1)
self.rpe = nn.Parameter(torch.zeros(3, num_heads, hops.max()+1))
pos_emb  = self.rpe[:, :, self.hops]             # forward 时查表 → (3,H,V,V)
w1       = self.fc1[i] + pos_emb[i]              # A + B (论文 Eq.5)
```

**(b) Dynamic Topological Encoding** —— 解决 P1（动作级），论文 Sec.3.2.2 / Eq.(3-4)

$$C = f_\theta\left( \Psi^0\big(\mathcal{D}_1^0, \mathcal{D}_2^0, \dots, \mathcal{D}_p^0\big) \right)$$

- 对每个动作序列：用 **关节对欧氏距离** $w_{ij} = \|x_i - x_j\|_2$ 构造动态加权图 $\mathcal{G}_D$
- **Graph Filtration**（按距离阈值由小到大依次加边）+ **Vietoris-Rips Complex** → 抽象单纯复形
- 用 **Persistent Homology** 提取 barcodes（连通分量 / 孔洞 的"生灭时间"）
- 可微 vectorization $\Psi^0$ 投影到 64 维 → 再用线性层 $f_\theta$ 投到每层 hidden dim

代码：`model/BlockGCN.py: Topo` + `TopoTrans`
```python
class Topo(nn.Module):
    def forward(self, x):
        x = x.mean(1)                            # 沿 person 维平均
        x = x.unsqueeze(-1) - x.unsqueeze(-2)    # 关节对差向量
        x = x.mean(-3); x = self.L2_norm(x)      # 距离矩阵 w_ij
        x = (x - x.min()) / (x.max() - x.min())  # 归一化到 [0,1]
        x = self.vr(x)                           # VietorisRipsComplex
        x = make_tensor(x).to(device)
        x = self.pl(x)                           # StructureElementLayer = Ψ⁰
        return x                                  # (N, 64)
```

**注入方式**：每层都把同一份 64 维 barcode 投到该层通道维，沿 (T, V) broadcast 加到主特征上。

#### 4.2.2 BlockGC —— 解决 P2，论文 Sec.3.3 / Eq.(6)

把 W 设计成**块对角矩阵**，等价于在通道维度上把 $d$ 切成 $K$ 组，每组独立做 GC：

$$H^{(l)} = \sigma\left( \begin{bmatrix} (A_1+B_1)(H_1+C_1) \\ \vdots \\ (A_K+B_K)(H_K+C_K) \end{bmatrix} \begin{bmatrix} W_1 & & \\ & \ddots & \\ & & W_K \end{bmatrix} \right)$$

| 方法 | 复杂度 | 参数 |
|---|---|---|
| Vanilla GC | $\mathcal{O}(\|V\|d^2)$ | $d^2 + \|V\|^2$ |
| Ensemble of GCs | $\mathcal{O}(K\|V\|d^2)$ | $Kd^2 + K\|V\|^2$ |
| **BlockGC** | $\mathcal{O}(\|V\|d^2/K)$ | $d^2/K + K\|V\|^2$ |

代码实现极其简洁 —— 用 `nn.Conv2d` 的 `groups` 参数：

```python
self.num_heads = 8 if in_channels > 8 else 1           # K = 8
self.fc1 = nn.Parameter(torch.eye(...).expand(3, K, V, V))  # 可学习 A_k
self.fc2 = nn.ModuleList([
    nn.Conv2d(in_c, out_c, 1, groups=self.num_heads)   # ← BlockGC 的 W
    for _ in range(3)
])
```

`groups=K` 让 Conv1×1 在通道维分组，每组只与自己组内的输入相连 —— 这就是块对角结构。

### 4.3 最终公式 (论文 Eq.5)

$$\boxed{H^{(l)} = \sigma\Big( (A^{(l)} + B^{(l)})\,(H^{(l-1)} + C^{(l)})\, W^{(l)} \Big)}$$

| 符号 | 含义 | 代码 |
|---|---|---|
| $A^{(l)}$ | 可学习邻接 | `self.fc1[i]` |
| $B^{(l)}$ | Static topo encoding (SPD 查表) | `self.rpe[:, :, self.hops]` |
| $H^{(l-1)}$ | 主特征 | `x` |
| $C^{(l)}$ | Dynamic topo encoding (PH barcode) | `self.t_i(self.topo(joint))` |
| $W^{(l)}$ | 块对角投影 | `self.fc2[i]` (`groups=K`) |

### 4.4 整体架构 (论文 Fig.4b)

```
Input (N, 3, 64, 25, M)
   ├── Topo (PH) → barcode (N, 64) ──┐
   │                                  │ 每层注入
   └── Linear + PE + BN               │
       (N·M, 128, 64, 25)             │
            │                         │
   l1 (BlockGC + MS-TCN, 128) + t0(a) ◄┘
   l2 (128) + t1(a)
   l3, l4 (128)
   l5 (128→256, stride=2 时间下采样)
   l6, l7 (256)
   l8 (256, stride=2)
   l9, l10 (256)
            │
   GAP over (T, V, M) → (N, 256)
   Linear → Softmax → 60/120/10 类
```

10 层 `TCN_GCN_unit` 堆叠 = `unit_gcn`(BlockGC + Static Topo) + `MultiScale_TemporalConv`。

### 4.5 数据流（4 模态融合）

| 模态 | Feeder 内做什么 | 用途 |
|---|---|---|
| **Joint (J)** | 减 spine center，保留 spine 轨迹 | 主输入 + Topo 分支永远用它 |
| **Bone (B)** | `bone[v1] = joint[v1] - joint[v2]` | 父子关节差向量 |
| **Joint Motion (JM)** | `data[1:] - data[:-1]` | 时间差分 |
| **Bone Motion (BM)** | B 模态再做时间差分 | |

4 个模态分别训练 4 个权重，最后由 `ensemble.py` 在 softmax score 上加权求和 —— 这就是论文 Tab.2 的 "4-Stream fusion"。

### 4.6 训练设置

| 超参 | 值 |
|---|---|
| Loss | **CrossEntropy**（论文明确强调不用 contrastive / language supervision） |
| Optimizer | SGD + Nesterov, momentum 0.9 |
| weight_decay | 4e-4 (NTU), 2e-4 (UCLA) |
| base_lr | 0.05, ×0.1 at epoch 110 / 120 |
| batch_size | 64 (NTU), 16 (UCLA) |
| window_size | 64 帧 |
| epoch | 110~140 |

---

## 五、实验结果（占位，等截图）

### 5.1 论文 SOTA 表格 (Tab.2)

| 数据集 / 划分 | 模态 | BlockGCN | 上一 SOTA (FR Head) |
|---|---|---|---|
| NTU60 X-Sub  | 4-Stream | **93.1** | 92.8 |
| NTU60 X-View | 4-Stream | **97.0** | 96.8 |
| NTU120 X-Sub | 4-Stream | **90.3** | 89.5 |
| NTU120 X-Set | 4-Stream | **91.5** | 90.9 |
| NW-UCLA      | 4-Stream | **96.9** | 96.8 |

参数量 **1.3M**，比 FR Head 少 35%。

### 5.2 复现结果（待填）

| 数据集 / 划分 | 模态 | 论文报告 | 本次复现 | 差距 | 备注 |
|---|---|---|---|---|---|
| NTU60 X-Sub | J | 90.9 | ? | | tensorboard 截图位 |
| NTU60 X-Sub | B | - | ? | | |
| NTU60 X-Sub | JM | - | ? | | |
| NTU60 X-Sub | BM | - | ? | | |
| NTU60 X-Sub | **4-Stream** | 93.1 | ? | | `ensemble.py` 输出 |
| UCLA | J | 95.5 | ? | | |

### 5.3 训练过程截图建议

- **训练曲线**：`tensorboard --logdir work_dir/.../runs` 截 acc / loss / lr 三张曲线
- **混淆矩阵**：`work_dir/.../epoch*_test_each_class_acc.csv` 用 matplotlib 画 heatmap
- **学习率衰减**：取 epoch=109/110/119/120 附近的 lr 曲线，可见阶梯式下降
- **参数量 vs 精度散点图**：复用论文 Fig.1，把自己复现的点叠上去

### 5.4 消融实验（可选，对应论文 Tab.4）

| GC | BlockGC | PE | Dynamic | Static | Params | Acc(%) |
|---|---|---|---|---|---|---|
| ✓ | - | - | - | - | 2.1M | 85.2 |
| - | ✓ | - | - | - | 1.2M | 85.8 |
| - | ✓ | ✓ | - | - | 1.2M | 86.0 |
| - | ✓ | ✓ | - | ✓ | 1.2M | 86.2 |
| - | ✓ | ✓ | ✓ | - | 1.3M | 86.7 |
| - | ✓ | ✓ | ✓ | ✓ | **1.3M** | **86.9** |

> 若有时间可以自己跑前 2~3 行验证 BlockGC 单独的增益。

---

## 六、附录：踩坑速查表

| 报错关键字 | 章节 | 根因 |
|---|---|---|
| `No module named 'resource'` | 2.1 ① | Windows 没有该模块 |
| `BrokenPipeError` / DataLoader 反复重启 | 2.1 ② | `num_worker > 0` + Windows spawn |
| `prefetch_factor option could only be specified` | 2.1 ② | `num_worker=0` 时不能传 `prefetch_factor` |
| `'Model' object has no attribute 'module'` | 2.1 ③ | 单卡未用 `DataParallel`，原代码假设始终包装 |
| `Expected all tensors to be on the same device` | 2.1 ④ / 2.2 ② | `joint` 没搬 GPU / `make_tensor` 默认 CPU |
| `YAMLLoadWarning` | 2.1 ⑤ | PyYAML 5.1+ 强制传 Loader |
| UCLA 训练 loss 不下降 / shape 不对 | 2.2 ① | `TopoTrans` 硬编码 `repeat(2,1)` |
| `No module named 'torchpack.runner.hooks'` | 2.3 | 商汤内部依赖 |
| `CUDA error: invalid device ordinal` | 3.1 | shell 里写了 `--device 0 1` 但只有 1 张卡 |
