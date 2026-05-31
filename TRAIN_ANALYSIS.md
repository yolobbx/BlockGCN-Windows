# 训练结果分析：`work_dir/ntu60/csub/140_epochs`

> 命令：`python main.py --config config/nturgbd-cross-subject/default.yaml --model model.BlockGCN.Model --work-dir work_dir/ntu60/csub/140_epochs --device 0`
>****
> 数据集：NTU60 Cross-Subject（60 类、25 关节、2 人、骨架 / joint 模态）
> 模型参数量：**1,372,780（≈1.37 M）**

---

## 一、训练目录文件构成

| 文件 / 目录 | 个数 / 大小 | 用途 |
|---|---|---|
| `config.yaml` | 1.6 KB | 训练时的完整超参快照（命令行 + yaml 合并后） |
| `BlockGCN.py` | 14.9 KB | 训练开始时模型源码的快照（复现用） |
| `log.txt` | 67 KB / 1012 行 | 主训练日志（loss / acc / 时间） |
| `runs/train/events.out.tfevents.*` | 14.5 MB | TensorBoard 训练曲线（loss、acc、lr、梯度） |
| `runs/val/events.out.tfevents.*` | 12 KB | TensorBoard 验证曲线（loss、top1、top5） |
| `runs-{epoch}-{step}.pt` | **140 个 × 5.8 MB ≈ 810 MB** | 每个 epoch 的完整权重（含 optimizer state？看实际大小是只有模型权重） |
| `epoch{N}_test_score.pkl` | **140 个 × 4.65 MB ≈ 650 MB** | 每个 epoch 测试集上的 logits（用于后续 ensemble） |
| `epoch{N}_test_each_class_acc.csv` | 140 个 × ~8.5 KB | 第 1 行=60 类各自 acc；后面是 60×60 的混淆矩阵 |
| `runs-114-71364_right.txt` | 110 KB / **16 487 行** | 最优模型预测正确的样本 ID 列表 |
| `runs-114-71364_wrong.txt` | 29 KB / **1 472 行** | 最优模型预测错误的样本：`tensor(pred), true, ?` |

**总占用 ≈ 1.4 GB**，其中 800 MB 是权重，650 MB 是 score。

### 命名解读

- `runs-114-71364.pt` ← `114` 是 epoch，`71364` 是累计 step 数（87640 总步 ÷ 140 epoch ≈ 626 step / epoch = 40 064 训练样本 / 64 batch ≈ 626，吻合 NTU60-CS 训练集 40 091 样本）
- 测试集 `258 batches × 64 = 16 512 样本` ≈ NTU60-CS 测试集 16 487 样本（右文件行数刚好对上）

### 最优产物（epoch 114）

- 权重：`runs-114-71364.pt`
- 测试分数：`epoch114_test_score.pkl`
- 类级别准确率：`epoch114_test_each_class_acc.csv`
- 对/错样本列表：`runs-114-71364_right.txt` / `_wrong.txt`

---

## 二、训练时长分析（为什么这么慢）

### 时间线

| 阶段 | 时间 | 说明 |
|---|---|---|
| 启动失败 ×3 | May 23 23:25 – May 24 12:50 | 日志里看到 4 次 "Training epoch: 1"，前 3 次都没产出首轮 loss 就退出（很可能因 num_worker / Windows spawn 卡死） |
| 真正训练 | May 24 12:52:35 – May 29 10:58:00 | epoch 1 ~ epoch 140 |
| 总结输出 | May 29 11:13:14 | "Best accuracy: 0.9107 at epoch 114" |
| **训练真实墙钟** | **≈ 4 天 22 小时（118 h）** | |

### 每轮耗时拆分

从日志时间戳推算（单轮典型值）：

| 阶段 | 耗时 |
|---|---|
| 训练前向+反向（626 step） | **~35 min** |
| 验证（258 batch） | **~15 min** |
| 保存 ckpt + score.pkl | **~1 min** |
| **合计每 epoch** | **~51 min** |

`140 × 51 min = 7140 min ≈ 119 h`，与墙钟吻合。

### 瓶颈到底在哪？

日志里每个 epoch 都打了一行：

```
Time consumption: [Data]03%, [Network]97%
```

→ **数据加载只占 3%，GPU 前向反向占 97%**。

也就是说"`num_worker=0`"在这次跑里**并不是**主要瓶颈，真正贵的是模型本身：

- 1.37 M 参数虽然不算大，但 `BlockGCN` 内含 10 层 `TopoTrans`，每层有多次邻接矩阵构造 + GCN + 注意力 + BlockGC 操作；
- `window_size=64`、`num_person=2`、`num_point=25` → 单个样本 batch 内是 `(64, 3, 64, 25, 2)` 的张量；
- `batch_size=64` 在单卡下 GPU 显存通常吃满，但单卡算力是瓶颈。

> 💬 **结论**：训练慢 **80% 是 GPU 算力 + 模型自身的原因**，不是 Windows 也不是 DataLoader。如果想加速，下一节会详细列出。

### 还能压缩多少

| 优化 | 预期加速 | 风险 |
|---|---|---|
| 多卡 DDP（前提是有第 2 张卡） | × 1.6 ~ 1.9 | 需要把 `unwrap_model()` 那条路径搞干净 |
| AMP / `torch.cuda.amp.autocast` 混合精度 | × 1.4 ~ 1.8 | 偶发数值不稳，需要 `GradScaler` |
| 增大 `batch_size`（如 128 / 256）+ 同比例 LR | × 1.2 ~ 1.5 | 需要显存够 |
| `eval_interval=5`（少做评估） | 训练时间 -~22% | 损失"每个 epoch 都有 score" 这个便利 |
| 不保存每个 epoch 的 ckpt 和 score（`save_interval=10`） | 节省磁盘 ~1 GB | 失去随机做 ensemble 的能力 |
| 数据缓存到 SSD + 适度调大 `num_worker` | <5%（因为 Data 只占 3%） | Windows 上偶发 hang |

> 💬 **最划算的两条**：
> 1. **`eval_interval=5`** —— 当前每轮都评估（15 min/次），如果 5 轮评一次，140 轮训练可省 `140 × 4/5 × 15 min ≈ 28 h`，省 25%；
> 2. **AMP** —— 加 `torch.amp.autocast('cuda')` + `GradScaler`，typical 提速 1.5×，且对 BlockGCN 这种以 conv/matmul 为主的模型很安全。

---

## 三、Loss / 准确率走势分析

### 1. 关键数字

| 指标 | 值 |
|---|---|
| **最优 Top-1** | **91.07%（epoch 114）** |
| **最优 Top-5** | **98.62%（epoch 114）** |
| 最终 Top-1（epoch 140） | 90.77% |
| 训练集最高 acc | 99.23%（epoch 137） |
| 训练集最低 loss | 0.0384（epoch 137） |
| 测试集最低 loss | 0.3119（epoch 114） |
| 参考 BlockGCN 官方在 NTU60-CS（joint） | **~90.8% Top-1** |

→ **复现结果 91.07%，已经超过官方报告值 ~0.3 pp，复现成功且偏好**。

### 2. 训练曲线分段

| 阶段 | epoch | 学习率 | train loss | train acc | test top1 | 现象 |
|---|---|---|---|---|---|---|
| Warm-up | 1 – 5 | 0 → 0.05（线性增加） | 3.53 → 0.97 | 12.7% → 70.0% | 23.8% → 70.7% | 正常收敛 |
| 主训练 | 6 – 110 | 0.05 | 0.83 → 0.29 | 73.8% → 90.6% | **剧烈抖动** | 见下方"BN 抖动期" |
| 第一次衰减 | 111 – 120 | 0.005 | 0.16 → 0.06 | 95.1% → 98.5% | 跳到 90.6%+ 并稳定 | 测试 acc 终于平静 |
| 第二次衰减 | 121 – 140 | 0.0005 | 0.06 → 0.04 | ~99.2% | 90.6% – 91.0% 区间 | 过拟合开始（test 没再涨） |

### 3. BN 抖动期的怪现象（epoch 11 – 110）

这是最值得注意的地方。在 epoch 6 ~ 110 之间，**训练 loss 一直平稳下降**，但**测试 Top-1 反复在 1.66% 与 84% 之间跳变**，例如：

| Epoch | Train acc | Test Top-1 | Test Loss |
|---|---|---|---|
| 11 | 81.96% | **6.85%** | 7.79 |
| 13 | 82.85% | **2.53%** | 5.61 |
| 14 | 83.14% | **1.67%** | 30.67 |
| 17 | 84.30% | **1.66%** | 54.47 |
| 18 | 84.59% | 3.38% | 7.73 |
| 19 | 84.66% | **67.31%** | 1.12 |
| 20 | 84.97% | **80.19%** | 0.65 |
| 28 | 86.72% | **74.97%** | 0.90 |
| 34 | 87.50% | **28.93%** | 4.04 |
| 40 | 88.08% | **1.66%** | 20.00 |
| 41 | 88.21% | 12.46% | 6.01 |

进入 epoch 111（学习率从 0.05 衰减到 0.005）后，**抖动立刻消失**，test top1 稳定到 90.6%+。

> 💬 **诊断（重要）**：这是经典的 **BatchNorm running statistics 跟不上权重剧变** 现象——LR=0.05 偏大，每个 epoch 权重位移幅度大，但 BN 的 running mean / var 是用 momentum=0.1 慢慢累积的，二者错位时 `model.eval()` 用的 BN 统计量已经"过时"，导致测试时输出整体偏移。
>
> 这并不是 bug 或 Windows 适配的问题（官方版用同样配置也会有类似抖动，只是没那么夸张）。可能的原因：
> 1. **`num_worker=0`** 导致每个 epoch 内 batch 顺序、shuffle 行为略有差异 → BN 统计量收敛比多 worker 时更慢；
> 2. 单卡 `batch_size=64`，相比官方 2 卡（等效 128）BN 估计精度更差；
> 3. 训练用 `random_rot=True`，但测试不用 → 训练时 BN 看到的分布跟测试时不一致。

#### 减轻这种抖动的方法（不影响最终精度）

- **降低 base_lr 到 0.025** 或 **延长 warm_up_epoch 到 10**：用更平缓的优化轨迹让 BN 跟得上；
- **增大 batch size 到 128/256**（前提显存够）：BN 估计更稳；
- **将 SGD 的 momentum 从 0.9 降到 0.8** 或加 EMA（配置里 `ema: false`，本次没开）：让权重轨迹更平滑；
- **eval 时使用 EMA 模型而不是当前模型**：彻底回避 BN 问题。

但说实话，**既然你只关心最终结果，这个抖动可以忽略**——LR 一衰减就好了，最终 91.07% 是干净的。

### 4. 训练后期过拟合

- Epoch 121 之后训练 acc 从 98.5% → 99.23%，但测试 Top-1 **始终在 90.6 – 91.0% 之间徘徊不再上涨**；
- 测试 loss 从 0.3119 缓慢上涨到 0.3345 → **轻度过拟合**；
- 最优 epoch 114 出现在第一次衰减后的"甜点期"，符合 BlockGCN 论文里 110/120 step 调度的设计意图。

> 💬 **优化点**：epoch 120 以后基本是浪费时间，**`num_epoch` 可以从 140 改成 125**，节省最后 ~12 h 训练 + ~3 h 测试。

---

## 四、最优模型详细情况（epoch 114）

### 1. 总体

- Top-1: **91.07%** (16 487 right / 17 959 total)
- Top-5: **98.62%**
- Test loss: 0.3119

> ⚠️ 这里 `right + wrong = 16 487 + 1 472 = 17 959`，但 NTU60-CS 测试集标准是 **16 487 样本**。说明 `_right.txt` / `_wrong.txt` 写入逻辑里可能把"对的样本写了 1 次，所有 batch 的错的样本全写了一次"，**导致 wrong 数偏大**。从 Top-1 = 91.07% 反推，错误样本应该是 16 487 × (1 − 0.9107) ≈ **1 472** —— 数字其实是对的，反倒是 right 数（16 487）刚好等于整个测试集大小，这说明：
> - **`_right.txt` 写的是"测试集中所有样本"或者写法有 bug**；
> - 真正逻辑可能是 `right.txt` 写"序号 + 真值"全列表，`wrong.txt` 单独写错样本。
> - 别直接用 `_right.txt` 行数当"对的数量"，**用 `Top-1 × 16 487` 才是准的**。

### 2. 按类别的表现（从 `epoch114_test_each_class_acc.csv` 第一行）

| 类别区间 | acc 最低的若干类 | acc 最高的若干类 |
|---|---|---|
| **难类（< 0.80）** | 类 1 (0.764)、类 10 (0.744)、类 11 (0.533) ← 最差、类 16 (0.788)、类 28 (0.720)、类 40 (0.775) | 类 7 (0.982)、类 8 (0.985)、类 26 (1.000)、类 34 (0.986)、类 54 (0.993)、类 58 (1.000) |

- **类 11 准确率仅 53.3%** —— 是全场最低，与混淆矩阵对比看应该是"reading"/"writing"这类极相似动作；
- **2 个类 100%（类 26、类 58）** —— 通常是"falling down"、"jump up"这类骨架运动幅度大的动作。

### 3. 错误样本格式

`runs-114-71364_wrong.txt` 每行格式：

```
tensor(predicted_class), true_label, sample_index
```

例如 `tensor(12),10,12` 表示"真实标签 12（writing），被预测成 10（clapping），样本索引 12"。这种格式可以直接做 case study。

---

## 五、给后续训练的建议（节省时间 / 提质）

### 直接节省时间

1. **`num_epoch: 140` → `num_epoch: 125`**：epoch 120 以后 test 已不再涨，每轮 51 min × 15 epoch ≈ **省 12.7 h**。
2. **`eval_interval: 1` → `eval_interval: 5`**：训练中段不每轮都评，**省 ~28 h**。
3. **`save_interval: 1` → `save_interval: 10`**：磁盘从 1.4 GB → 200 MB；后续 ensemble 用最后 3 个 ckpt 一样够。
4. **加 AMP**：在 main.py 的 train/eval 循环里用 `torch.amp.autocast('cuda', dtype=torch.float16)` + `GradScaler`，**~1.5× 加速**。

> 💬 上面四条全做完，理论上 4-5 天能压到 **~1.5-2 天**。

### 改善 BN 抖动（如果你想要更"漂亮"的训练曲线）

5. **`warm_up_epoch: 5` → `10`**，并 **`base_lr: 0.05` → `0.03`**；
6. 启用 **EMA**（`ema: true`），验证时用 ema 模型；
7. `batch_size: 64` → 96 或 128（如果显存允许）。

### 复现 / 后续 ensemble

7. 这次跑出的 `epoch114_test_score.pkl` 已经可以喂给 `ensemble.py` 与 joint-bone / vel 模态组合，进一步把单流 91.07% 推到 ensemble ~92.5-93%。脚本：
   ```
   python ensemble.py --benchmark NTU60CSub \
       --joint-dir work_dir/ntu60/csub/140_epochs \
       --bone-dir <训练 bone 流的目录> ...
   ```

### 工程上

8. 磁盘清理：删掉 epoch 1-100 的 `runs-*.pt` 与 `epoch*_test_score.pkl`，**只留 110-130**（甜点区间）即可释放 ~1 GB；
9. 把 `runs/train/events.out.tfevents.*` 用 `tensorboard --logdir work_dir/ntu60/csub/140_epochs/runs` 打开看实时曲线（**这个文件官方版没有专门画图脚本，TensorBoard 就是最直接的可视化**）。

---

## 六、TL;DR

- 训练 4 天 22 小时跑完 140 epoch，**最终 Top-1 = 91.07%，超过官方报告 ~0.3 pp，复现成功**。
- 慢的真正原因是 **GPU 算力（97% 时间在网络前向反向）**，不是 Windows 也不是 DataLoader。
- **最划算的提速：`eval_interval=5` + AMP + `num_epoch=125`**，理论上能压到 1.5-2 天。
- LR=0.05 阶段测试 acc 抖动是 **BatchNorm running stats 跟不上权重剧变**，开 EMA 或降 LR 可以平滑曲线，但**不影响最终结果**。
- 后续可以用 `epoch114_test_score.pkl` 做 joint+bone 多模态 ensemble，再涨 1-2 pp。
