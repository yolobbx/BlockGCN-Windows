# BlockGCN (CVPR'24) vs CTR-GCN (ICCV'21)：差异、取舍与改进动机

两篇论文血缘很清晰——BlockGCN 在 `Implementation Details` 里说 "Our implementation builds upon the official code [2]"（[2] 即 CTR-GCN），但它对 CTR-GCN 的核心机制做了一次相当激进的"拆—换"。下面按"整体差异 / 舍去 / 改进 / 为什么"四条线讲。

## 1. 整体定位差异

| | CTR-GCN (ICCV21) | BlockGCN (CVPR24) |
|---|---|---|
| 核心叙事 | 共享一张拓扑不够，要 **每个通道学一张动态拓扑** | 可学拓扑会"灾难性遗忘"骨架先验，要 **把拓扑信息显式编码注回去**，同时砍掉冗余 |
| 空间块结构 | 3 个并行 CTR-GC 分支求和 | 1 个 BlockGC（块对角权重） |
| 拓扑来源 | A（共享、可学）⊕ Q（输入相关、逐通道 MLP 推出） | A（可学）⊕ B（SPD 查表静态编码）⊕ C（持续同调动态编码，加到特征上） |
| 参数 / FLOPs (4-stream) | 1.5M / 1.97G，X-Sub120 = 88.9 | 1.3M / 1.63G，X-Sub120 = 90.3 |

## 2. BlockGCN 从 CTR-GCN 里"舍去"了什么

1. **逐通道动态相关 Q（CTR-GC 的灵魂）**。CTR-GC 用 φ、ψ 把 (x_i, x_j) 投影成 N×N×C' 的 channel-specific correlation，再加到共享 A 上得到 R∈ℝ^(N×N×C')。BlockGCN 整个把它砍了——理由见下面 P1。
2. **每个 spatial module 里的 3 路并行 GC 分支**（CTR-GCN Fig.3a 上半部分的 3 个 CTR-GC + 求和）。BlockGCN 一个 block 里只剩一条 BlockGC。
3. **完整的 d×d 投影权重 W**，替换成 K 路块对角的 W₁…W_K（每块 d/K × d/K）。
4. **CTR-GCN 的"通道维拼接式聚合"** Z = [R₁x̃_{:,1} ‖ … ‖ R_{C'}x̃_{:,C'}]——逐通道一张图——这种细粒度通道图整体被抛弃。

## 3. BlockGCN 新引入 / 改进的东西

1. **静态拓扑编码 B**：用关节对在骨架图上的最短路径距离 d_{ij} 作为索引，从一张小的 embedding table 里查出标量 B_{ij}，加到 A 上。
2. **动态拓扑编码 C**：对每段输入序列建一个用欧氏距离加权的动态图，跑 Vietoris-Rips 滤波，提 0/1-维 barcode，可微分向量化后投到隐层维度，**加到 hidden features H 上**（不是加到 A 上）。
3. **BlockGC**：把通道分 K 组，A 也分 K 张（A_k），但 W 写成块对角——一次卷积里同时实现"多关系建模"和"参数减半"，复杂度 O(|V|d²/K) 对比 CTR-GCN 的 ensemble O(K|V|d²)。
4. 最终 spatial 算子：

   H = σ((A + B)(H + C) W_block)

## 4. 为什么这样改

**P1（动机一）：可学 A 会灾难性遗忘骨架先验。**
作者用两个证据说服读者：
- (i) 把 CTR-GCN / DecouplingGCN 的 A 初始化换成单位阵 / 全 1 / Kaiming uniform，精度几乎不变（Tab.3，差值 ≤0.2）——说明 CTR-GCN 训练完根本没在用骨架结构；
- (ii) 可视化显示训练后每层 A 之间彼此差别极大，且都远离 bone connection。

Q 即便存在也只是基于特征算"相似度"，并不重新注入物理连接。所以 BlockGCN 不再指望 A 自己记住骨架，而是用一个**离散、按 SPD 索引、不会被覆盖**的 embedding 表把拓扑硬塞回去。持续同调那一路则补上"A 本来想动态描述但描述不准的"全局结构特征（连通分量、洞），而且是注入到 feature 空间，避开了再次让 A 学这件事。

**P2（动机二）：多关系建模在 CTR-GCN 里是 O(K|V|d²) 的浪费。**
CTR-GCN 的 3 分支 ensemble 每条都带一个完整 d×d 的 W；BlockGCN 的 Tab.6 显示：
- DecouplingGCN 用 K 张 A 但共享一个完整 W，精度也只小涨；
- vanilla GC 用一个 W、参数 2.1M；
- 把 W 切成块对角（参数降到 1.2M，–43%）反而涨 0.6%。

作者的关键观察是 W 里有大量冗余——**不同语义关系本来就应该作用在解耦的特征子空间上**，块对角天然实现了"分组、独立、不互相污染"，而 ensemble 是反复在同一个完整特征空间上叠 K 套 W。

**为什么用持续同调而不是继续走 attention / Q 的路。**
CTR-GCN 系的动态拓扑都是 pair-wise 特征相似度，本质是局部的，难以表达"这只手和那只脚此刻是不是属于同一连通块"这种粒度的拓扑事件。barcode 在多个尺度阈值上同时观察图，能给出 inter-action 相似、intra-action 一致的描述（论文 Fig.6 的"梳头" vs "握手"对比）。这是 Q 提供不了的信息层级。

## 一句话总结

CTR-GCN 的思路是"让拓扑变细——每通道、每样本都不同"；BlockGCN 反过来认为这条路在工程上把骨架先验丢了、在结构上又造成 W 重复。它的对策是：**拓扑先验从 A 里搬出来用查表 / 持续同调显式编码**，把 A 解放成"剩下的可学部分"；同时把 ensemble + 完整 W 替换成 1 条 BlockGC + 块对角 W，用解耦的特征子空间承担多关系建模。结果是参数 1.3M < CTR-GCN 1.5M，X-Sub120 88.9 → 90.3。
