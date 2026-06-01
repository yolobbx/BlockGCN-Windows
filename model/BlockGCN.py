# ============================================================================
# BlockGCN: Redefine Topology Awareness for Skeleton-Based Action Recognition
# (CVPR 2024)
#
# 本文件实现了论文的两大核心创新：
#   1. Topological Encoding —— 解决可学习邻接矩阵"灾难性遗忘"骨骼拓扑的问题
#      · Static Topological Encoding   (静态)：基于最短路径距离 (SPD) 的位置编码
#        —— 对应代码里 unit_gcn.rpe / unit_gcn.hops，论文 Eq.(2) B_ij = e_{d_ij}
#      · Dynamic Topological Encoding  (动态)：基于 Persistent Homology 的拓扑描述
#        —— 对应代码里 Topo (VR 复形 + Barcode 向量化) + TopoTrans (映射到隐藏空间)
#        论文 Eq.(4): C = f_θ(Ψ⁰(D_1, ..., D_p))
#   2. BlockGC —— 用块对角权重矩阵替换 vanilla GC 的全连接权重
#      —— 对应代码里 unit_gcn.fc2 = Conv2d(..., groups=num_heads)
#      论文 Eq.(6)：把特征维度切分成 K 组，组内独立做 GC，组间不耦合
#
# 最终 spatial aggregation 公式 (论文 Eq.(5)):
#   H^(l) = σ( (A^(l) + B^(l)) (H^(l-1) + C^(l)) W^(l) )
#   其中 A 是可学习邻接矩阵，B 是 static encoding，C 是 dynamic encoding，W 是块对角权重
# ============================================================================

import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange, repeat
import torch.nn.functional as F
# torch_topological 是计算持续同调 (Persistent Homology) 的工具包
# VietorisRipsComplex: 由点对距离构造 VR 复形
# StructureElementLayer: 可微分地把 Barcode 向量化 (论文 Ψ⁰)
from torch_topological.nn.data import make_tensor
from torch_topological.nn import VietorisRipsComplex
from torch_topological.nn.layers import StructureElementLayer

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class MultiScale_TemporalConv(nn.Module):
    """
    多尺度时序卷积 (Multi-Scale TCN)。对应论文 Sec.3.4 "multi-scale temporal convolution module"。
    把通道切分成 6 个 branch：
      · 4 个 (1x1 + TemporalConv) 分支，分别用 dilation=[1,2,3,4]，捕捉不同时间尺度
      · 1 个 MaxPool 分支
      · 1 个 1x1 残差分支
    最后沿通道维 concat。每个 GCN block 后接一个 MultiScale TCN，串联成 spatial-temporal block。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=False,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class unit_gcn(nn.Module):
    """
    BlockGCN 的核心：BlockGC + Static Topological Encoding。

    论文公式 (Eq.5, Eq.6) 的实现：
        H^(l) = σ( Σ_k (A_k + B_k) · (H_k^(l-1) + C_k^(l-1)) · W_k^(l) )

    关键点：
    1. multi-relational (论文 K 组语义关系)
       这里同时通过两种方式实现：
       a) 用 3 个并行 GC 分支 (i=0,1,2)，对应 spatial graph 的 self/inward/outward 三种关系
       b) 在每个分支内部把 channel 分成 num_heads (=8) 组，每组配独立的邻接矩阵 head
          —— 这就是论文里的 "K 组"，参数 fc1 形状 (3, num_heads, V, V)

    2. BlockGC (论文 Sec.3.3)
       特征投影 fc2 用 nn.Conv2d 的 `groups=num_heads`：
         W 在结构上是块对角矩阵 (block diagonal)，d/K × d/K 的小块沿对角线排列。
         参数量从 O(d²) 降到 O(d²/K)，且不同 group 的特征通道之间相互解耦，
         每组建模一种独立语义。

    3. Static Topological Encoding (论文 Sec.3.2.1，Eq.2)
       self.hops: 预先计算 SPD (Shortest Path Distance) 矩阵，hops[i,j] = i,j 之间最短路径长度。
       self.rpe : 形状 (3, num_heads, max_hop+1)，可学习的距离 → embedding 表 E。
       前向时 pos_emb = self.rpe[:, :, self.hops] 把 hops 当索引去查表得到 B_ij。
       由于只有 embedding 表本身在训练，bone 距离关系本身不会被遗忘 —— 这就是论文
       要解决的 catastrophic forgetting of skeletal topology 的核心机制。

    Args:
        A : 3 个先验邻接矩阵 (self/inward/outward)，形状 (3, V, V)。用于初始化 + 推 hops。
        adaptive / alpha : 兼容老接口（本实现里 A 已被替换为可学习 fc1，不再需要）。
    """

    def __init__(self, in_channels, out_channels, A, adaptive=True, alpha=False):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        # K = num_heads：BlockGC 的分组数；输入太小时退化为单组
        self.num_heads = 8 if in_channels > 8 else 1

        # fc1: 论文里的可学习邻接矩阵 A_k —— shape (3, num_heads, V, V)
        # 3 = 3 个并行 GC 分支 (self/inward/outward)
        # num_heads = K = 论文里的分组数
        # 每个 head 自己一个 V×V 邻接，初始化为单位阵 (相当于先验"自连"——后面靠 rpe 注入拓扑)
        self.fc1 = nn.Parameter(torch.stack([torch.stack([torch.eye(A.shape[-1]) for _ in range(self.num_heads)], dim=0) for _ in range(3)], dim=0), requires_grad=True)

        # fc2: 论文里的 W_k —— 块对角权重矩阵 (BlockGC)
        # 关键：groups=num_heads 让 Conv 沿通道维分组，每组只与自己组内的输入相连
        # 这等价于一个 d × d 的块对角矩阵，块大小为 d/K，整体参数量 d²/K
        self.fc2 = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1, groups=self.num_heads) for _ in range(3)])

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        # ====================================================================
        # 预计算 k-hop SPD 矩阵：self.hops[i, j] = 关节 i 到 j 在骨骼图上的最短路径长度
        # 这是 Static Topological Encoding (论文 Eq.2 d_{i,j}) 的核心。
        # 因为骨骼图是固定的，SPD 可以一次性算好，训练时只更新 embedding 表。
        # ====================================================================
        h1 = A.sum(0)            # 把 3 个先验邻接合并：哪些关节之间有边
        h1[h1 != 0] = 1          # 二值化

        # h[k] = 仅经过 k 步可达的节点 mask（先累积，再差分）
        h = [None for _ in range(A.shape[-1])]
        h[0] = np.eye(A.shape[-1])     # 0 步：自己
        h[1] = h1                       # 1 步：直接邻居
        self.hops = 0 * h[0]
        for i in range(2, A.shape[-1]):
            # 矩阵幂：从 i-1 步可达的节点再走 1 步
            h[i] = h[i-1] @ h1.transpose(0, 1)
            h[i][h[i] != 0] = 1

        # 取差分得到"恰好 i 步可达"的 mask，hops 累加距离值
        for i in range(A.shape[-1]-1, 0, -1):
            if np.any(h[i]-h[i-1]):
                h[i] = h[i] - h[i - 1]
                self.hops += i*h[i]
            else:
                continue

        self.hops = torch.tensor(self.hops).long()    # shape (V, V)，值 ∈ [0, max_hop]

        # rpe: SPD → embedding 表 E (论文 B_ij = e_{d_ij})
        # 3 个分支 × num_heads 个 head × (max_hop+1) 个可能的距离值
        # 训练只更新这个表，骨骼距离结构本身不会被遗忘
        self.rpe = nn.Parameter(torch.zeros((3, self.num_heads, self.hops.max() + 1,)))

        self.in_channels = in_channels
        self.hidden_channels = in_channels if in_channels > 3 else 64

        if alpha:
            self.alpha = nn.Parameter(torch.ones(1, self.num_heads, 1, 1, 1))
        else:
            self.alpha = 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def L2_norm(self, weight):
        # 沿倒数第二维 L2 归一化，让邻接矩阵尺度可控（防梯度爆炸、与 rpe 加和时尺度匹配）
        weight_norm = torch.norm(weight, 2, dim=-2, keepdim=True) + 1e-4  # H, 1, V
        return weight_norm

    def forward(self, x):
        # 输入: x ∈ (N, C, T, V)   N=batch*M, M=num_person
        N, C, T, V = x.size()
        y = None

        # ---- Static Topological Encoding 查表 ----
        # pos_emb[i, h, u, v] = self.rpe[i, h, hops[u, v]]
        # 形状: (3, num_heads, V, V)，是论文里的 B^(l) 矩阵
        pos_emb = self.rpe[:, :, self.hops]

        # ---- 3 个并行 GC 分支 (multi-relational, 类似 self/inward/outward) ----
        for i in range(3):
            # fc1[i] 形状 (num_heads, V, V) ——可学习邻接 A^(l)_i
            weight_norm = self.L2_norm(self.fc1[i])
            w1 = self.fc1[i]
            w1 = w1 / weight_norm

            # 把 A 和 B 相加 (论文 Eq.5 的 (A+B) 项)
            # 归一化后相加保证 SPD encoding 的影响始终不会被可学习 A 完全淹没
            w1 = w1 + pos_emb[i] / self.L2_norm(pos_emb[i])

            # ---- BlockGC 的 spatial aggregation ----
            # 把通道维 C 切分成 num_heads 组：(N, H, C/H, T, V)
            x_in = x.view(N, self.num_heads, C // self.num_heads, T, V)
            # einsum 在每个 head 内独立做 spatial 聚合: x' = x · w1
            # nhctv, hvw -> nhctw  即把节点维 v 通过邻接 w1[h, v, w] 聚合到 w
            z = torch.einsum("nhctv, hvw->nhctw", (x_in, w1)).contiguous().view(N, -1, T, V)

            # ---- BlockGC 的 feature projection (块对角 W_k) ----
            # fc2[i] 是 groups=num_heads 的 Conv2d，等价于块对角矩阵投影
            z = self.fc2[i](z)

            # 3 个分支 element-wise sum (论文 Fig.4a 的 Σ)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)        # 残差连接，保证训练稳定
        y = self.relu(y)
        return y

class TCN_GCN_unit(nn.Module):
    """
    BlockGCN 的基本构建块 (论文 Fig.4b)：
        x ─► unit_gcn (BlockGC + Static Topo Encoding) ─► MultiScale TCN ─► + residual ─► ReLU
    论文里 spatial GC 和 temporal CNN 交替堆叠，共 10 层。
    """
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2], num_point=25, num_heads=16, alpha=False):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, alpha=alpha)
        # self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)


    def forward(self, x):

            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))

            return y
        
        
class TopoTrans(nn.Module):
    """
    Dynamic Topological Encoding 的"投影/映射"模块。对应论文 Eq.(4) 中的 f_θ：
        f_θ : R^{|V| × d'} ─► R^{|V| × d}
    把 Topo 模块输出的 64 维 barcode 向量化结果，映射到每一层 GCN 的隐藏维度 d (128 或 256)，
    然后在 forward 时作为 C^(l) 加到隐藏特征上 (论文 Eq.5: H + C)。

    实现细节：
      · 输入 x 是 shape (N, 64) 的 barcode 向量化结果（整段序列共享一份动态拓扑描述）
      · repeat(num_person, 1) 把 N → N*M，以匹配后面 (N*M, C, T, V) 的特征形状
      · 最终输出 (N*M, out_dim, 1, 1)，会沿 T、V 维 broadcast 加到主分支特征上
    """
    def __init__(self, out_dim, num_person=2):
        super(TopoTrans, self).__init__()
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(64,out_dim)
        # self.tanh = nn.Tanh
        # self.pa = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.bn = nn.BatchNorm1d(out_dim)
        self.num_person = num_person
        
    def forward(self, x):
        x = x.repeat(self.num_person,1)
        # x = x.repeat(1,1)
        # x = x
        x = self.mlp(x)
        #BN
        x = self.bn(x)
        x = self.relu(x)
        
        return x.unsqueeze(2).unsqueeze(3)


class Topo(nn.Module):
    """
    Dynamic Topological Encoding —— 用持续同调 (Persistent Homology) 提取动态拓扑特征。
    对应论文 Sec.3.2.2 与 Eq.(3-4)。

    流程 (论文 Fig.4a 左下):
        输入 pose sequence (joint 坐标)
            │  ① 用 关节对欧氏距离 w_ij = ||x_i - x_j||_2 构造动态加权图 G_D
            ▼
        Graph Filtration ε_1 ⊆ ε_2 ⊆ … ⊆ ε_m   (按距离阈值由小到大不断"长出"边)
            │  ② Vietoris-Rips Complex —— 把图升维成抽象单纯复形 K_i
            ▼
        计算 Barcodes (birth-death pairs) 描述连通分量 / 孔洞 的生灭时间
            │  ③ 可微 vectorization Ψ⁰ : 把 barcodes 投到 R^{V × d'}
            ▼
        输出 d' 维向量 (这里 d'=64)，每个动作样本独有 → "动态"拓扑描述符

    代码细节：
      · x.mean(1) : 沿 M (person) 维取平均，得到 (N, T, V, C)→(N, C, T, V) 的简化骨架
      · x.unsqueeze(-1) - x.unsqueeze(-2) : 计算关节对差向量
      · x.mean(-3) + L2_norm : 沿时间 T 平均 + 取模 → 关节对距离矩阵 w_ij (V × V)
      · (x - min)/(max - min) : 归一化到 [0, 1]，便于 VR 复形构造
      · self.vr(x) : 构造 VR complex，输出每个 persistence diagram
      · make_tensor : 把可变长度的 barcode 列表 pad 成定长张量
      · self.pl(x) : StructureElementLayer，可微地把 barcode 投影成 64 维特征
    """
    def __init__(self, dims=0):
        super(Topo, self).__init__()
        self.vr = VietorisRipsComplex(dim=dims)
        self.pl = StructureElementLayer(n_elements=64)
        self.relu = nn.ReLU()
    def L2_norm(self, weight):
        weight_norm = torch.norm(weight, 2, dim=1) # H, 1, V
        return weight_norm
   
    def forward(self, x):
        device = x.device
        x = x.mean(1)
        x = x.unsqueeze(-1) - x.unsqueeze(-2)
        x = x.mean(-3)
        x = self.L2_norm(x)
        x = (x-torch.min(x))/(torch.max(x)-torch.min(x))
        x = self.vr(x)
        x = make_tensor(x).to(device)
        x = self.pl(x)
        return x



class Model(nn.Module):
    """
    BlockGCN 完整模型 (论文 Fig.4b)。

    结构 (Joint 模态为例):
        Input: (N, C=3, T=64, V=25, M=2)  —— batch / 通道 / 时间 / 关节 / 人
            │
            ├── Topo(joint)            ─► barcode 向量化 a (N, 64)         # 动态拓扑 (整段序列共享一份)
            │
            └── linear embedding + pos_embedding + data_bn
                │
                ▼  (N*M, 128, T, V)
            l1 (+ t0(a))  ┐
            l2 (+ t1(a))  │ 4 层 128 维 spatial-temporal block
            l3 (+ t2(a))  │
            l4 (+ t3(a))  ┘
            l5 (+ t4(a))  ─► stride=2, 升维到 256，时间下采样
            l6 (+ t5(a))  ┐
            l7 (+ t6(a))  │ 3 层 256 维
            l8 (+ t7(a))  │
            l9 (+ t8(a))  │ stride=2，再下采样
            l10(+ t9(a))  ┘
                │
                ▼ Global Average Pooling over (T, V, M)
                ▼ Linear → Softmax (cross entropy)

    其中每个 TCN_GCN_unit 内部 = BlockGC (含 Static Topo Encoding) + MultiScale TCN。
    动态拓扑 a 通过 t_i (TopoTrans) 映射到每一层的 hidden dim，再加到隐藏特征上 (论文 Eq.5 的 H+C)。
    """
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_set=3, alpha=False, window_size=64, **kwargs):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        # A: 3 个先验邻接矩阵 (self_link / inward / outward)，仅用于推导 SPD / 提供形状
        A = self.graph.A  # 3, 25, 25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * 128 * num_point)

        # 把原始坐标 (C=3) 升维到 128，再加可学习的关节位置编码 (PE)
        # 对应论文 Tab.4 提到的 "learnable absolute positional embedding"
        self.to_joint_embedding = nn.Linear(in_channels, 128)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, 128))

        # 10 层 spatial-temporal block (论文 Sec.3.4: 堆叠 10 次)
        self.l1 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l2 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l3 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l4 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l5 = TCN_GCN_unit(128, 256, A,  stride=2, adaptive=adaptive, alpha=alpha)
        self.l6 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)
        self.l7 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)
        self.l8 = TCN_GCN_unit(256, 256, A, stride=2, adaptive=adaptive, alpha=alpha)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, alpha=alpha)

        # 每层一个 TopoTrans：把同一个 64 维 barcode 投到对应层的通道维 (128 / 256)
        # 论文 Tab.7 验证：在每一层都注入动态拓扑比只在首层注入效果好
        self.t0 = TopoTrans(out_dim=128, num_person=num_person)
        self.t1 = TopoTrans(out_dim=128, num_person=num_person)
        self.t2 = TopoTrans(out_dim=128, num_person=num_person)
        self.t3 = TopoTrans(out_dim=128, num_person=num_person)
        self.t4 = TopoTrans(out_dim=128, num_person=num_person)
        self.t5 = TopoTrans(out_dim=256, num_person=num_person)
        self.t6 = TopoTrans(out_dim=256, num_person=num_person)
        self.t7 = TopoTrans(out_dim=256, num_person=num_person)
        self.t8 = TopoTrans(out_dim=256, num_person=num_person)
        self.t9 = TopoTrans(out_dim=256, num_person=num_person)

        # 全局共享的 Persistent Homology 提取器
        self.topo = Topo()

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, y, joint):
        """
        Args:
            x     : 主输入特征 (N, C, T, V, M)
                    —— 对 joint 模态：去掉 spine center 后的相对坐标 + spine 自身轨迹
                    —— 对 bone 模态：父关节减子关节得到骨向量
                    —— 对 motion 模态：相邻帧做差
            y     : one-hot label (训练时传入用作 mmd loss 等的辅助；这里只是 pass-through 返回)
            joint : 原始关节坐标 (N, C, T, V, M)
                    —— 不论 x 是哪种模态，joint 始终是原始关节坐标，
                       因为 Persistent Homology 必须基于真实欧氏距离来算 (论文 w_ij = ||x_i - x_j||_2)
        """
        N, C, T, V, M = x.size()
        N, C, T, V, M = joint.size()

        # ---- 1) Dynamic Topological Encoding ----
        # 用 joint 坐标 (无论 x 是不是 joint 模态) 算 Persistent Homology
        a = rearrange(joint, 'n c t v m -> n m c t v', m=M, v=V).contiguous()
        a = self.topo(a)                                       # (N*M, 64) barcode 向量化结果

        # ---- 2) 输入 embedding ----
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        x = self.to_joint_embedding(x)                          # C=3 → 128
        x += self.pos_embedding[:, :self.num_point]              # 加关节位置编码 (PE)
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        x = self.data_bn(x)
        # 重整形状为 (N*M, 128, T, V)，后面所有 spatial-temporal block 都用这个形状
        x = x.view(N, M, V, 128, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, 128, T, V)

        # ---- 3) 10 层 ST-block，每层注入动态拓扑 C^(l) = t_i(a) ----
        # 注意：t_i(a) 形状 (N*M, C_l, 1, 1)，会沿 T、V 维 broadcast，相当于给每个关节/每帧都加上同一份"动作级"拓扑描述
        x = self.l1(x + self.t0(a))
        x = self.l2(x + self.t1(a))
        x = self.l3(x + self.t2(a))
        x = self.l4(x + self.t3(a))
        x = self.l5(x + self.t4(a))
        x = self.l6(x + self.t5(a))
        x = self.l7(x + self.t6(a))
        x = self.l8(x + self.t7(a))
        x = self.l9(x + self.t8(a))
        x = self.l10(x + self.t9(a))

        # ---- 4) 全局池化 + 分类 ----
        # x: (N*M, 256, T', V) → reshape 回 (N, M, 256, T'*V) → 沿 (T'*V, M) 求平均 → (N, 256)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x), y
