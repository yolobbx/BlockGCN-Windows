# BlockGCN Windows 适配版改动说明

本文档对比 `BlockGCN-main`（官方原版，Linux 环境）与 `BlockGCN`（Windows 适配 + 已跑通版），逐文件列出改动点、改动原因，并在文末给出仍可优化的方向。

---

## 一、整体差异概览

### 仅在 Windows 版（`BlockGCN/`）中存在的新增内容

| 路径 | 说明 |
|---|---|
| `requirements.txt` | 新增依赖清单（大多被注释，实际安装的是末尾若干行） |
| `config/ucla/default.yaml` | 新增 NW-UCLA 数据集配置（官方目录里没有这个 yaml） |
| `torchlight/__init__.py` | 新增顶层包导出，便于 `from torchlight import IO, ...` |
| `main copy.py` | `main.py` 的中间备份版本（与最终 `main.py` 略有差异，可删除） |
| `GCN.code-workspace` | VSCode 工作区文件，与训练逻辑无关 |
| `data/ntu60/`, `data/ntu120/raw_data/`, `data/NW-UCLA/all_sqe/` 等 | 实际已生成的数据/npz/raw 数据，原版只放了处理脚本 |
| `work_dir/`, `__pycache__/` | 训练产物与编译缓存，运行后自然产生 |

### 内容被修改的文件

| 文件 | 改动类型 |
|---|---|
| `main.py` | Windows 兼容 + DDP/单卡兼容 + DataLoader 兼容 + YAML 兼容 |
| `model/BlockGCN.py` | `num_person` 通用化（兼容 NTU 双人 / UCLA 单人）+ 设备一致性修复 |
| `torchlight/torchlight/util.py` | 缺失依赖（torchpack/PaviLogger）容错导入 |
| `train.sh` / `evaluate.sh` | 改成单卡命令 + 增加 UCLA 训练/测试示例 |

---

## 二、逐文件改动详解

### 1. `main.py` —— 改动最多，覆盖三类问题

#### (1) Windows 不支持 `resource` 模块
原版直接 `import resource` 并调用 `setrlimit` 提升打开文件数上限。Windows 上 `resource` 模块不存在，启动就会 `ImportError`。

```python
# 原版
import resource
...
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

# Windows 版
try:
    import resource
except ImportError:
    resource = None
...
rlimit = None
if resource is not None:
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
```
**原因**：兼容 Windows，让脚本能正常启动。

#### (2) 屏蔽冗余警告
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
```
**原因**：纯日志清爽化，不影响功能。

#### (3) `num_worker` 默认值由 16 改为 0
```python
# 原版
default=16,
# Windows 版
default=0,
```
**原因**：Windows 下 PyTorch DataLoader 多进程经常出现 `BrokenPipeError`、`spawn` 反复重启的问题；先设成 0（主进程加载）确保跑通。

> 💬 **留言（优化点）**：`num_worker=0` 在 Windows 上很稳但很慢，I/O 会成为瓶颈。建议在跑通后改成 `2`~`4` 试试，并把 `if __name__ == '__main__':` 守卫加在 `main.py` 入口（Windows `spawn` 必备）。可以在配置里通过 yaml 显式给出，避免每次手改。

#### (4) DataLoader 参数兼容 `num_worker=0`
原版无条件传 `prefetch_factor=16`，但 PyTorch 规定 `num_workers=0` 时不能传 `prefetch_factor`。Windows 版抽出 `loader_args` 字典动态拼装：

```python
loader_args = dict(
    num_workers=self.arg.num_worker,
    worker_init_fn=init_seed)
if self.arg.num_worker > 0:
    loader_args['prefetch_factor'] = 2
...
self.data_loader['train'] = torch.utils.data.DataLoader(
    dataset=Feeder(**self.arg.train_feeder_args),
    batch_size=self.arg.batch_size,
    shuffle=True,
    drop_last=True,
    **loader_args)
```
**原因**：让 `num_worker=0` 与 `>0` 两种情况都能跑。同时 `prefetch_factor` 由 16 改为 2，是更保守的默认值。

> 💬 **留言（优化点）**：`prefetch_factor` 给得太小会让 GPU 等数据；如果显存够、CPU 够，`num_worker=4, prefetch_factor=4` 一般更稳。

#### (5) 单卡 / 多卡（DDP）兼容
原版只在 `nn.DataParallel` 包装后跑，因此处处用 `self.model.module.num_class`；单卡 / 未包装时 `.module` 会报 `AttributeError`。Windows 版做了两件事：

```python
def unwrap_model(self):
    return self.model.module if hasattr(self.model, 'module') else self.model
```
并在调用处改为 `self.model.num_class`（注释里保留了 `unwrap_model()` 的写法作为备选）。

**原因**：单卡训练（`--device 0`）下消除 `.module` 报错。

> 💬 **留言（优化点）**：当前代码直接用 `self.model.num_class`，多卡 `DataParallel` 时这一句会报错。**更稳健的写法是统一用 `self.unwrap_model().num_class`**，单卡多卡都不用改。建议把所有 `self.model.num_class` 与 `self.model.module.num_class` 全部换成 `self.unwrap_model().num_class`。

#### (6) 显式把 `data` / `joint` 搬到 `output_device`
```python
data = data.to(self.output_device)
joint = joint.to(self.output_device)
```
**原因**：在 train/eval 主循环里多处补了显式 `.to(device)`，避免 `data` 在 CPU、`model` 在 GPU 导致的 `Expected all tensors to be on the same device` 错误。这通常是因为某些 Feeder 没在内部搬运张量。

> 💬 **留言（优化点）**：这种 `.to(...)` 散落在多处，并且 `joint.to(self.output_device)` 还在调用里又写了一遍（重复搬运）。建议在 Feeder/`collate_fn` 里统一搬到 device，或者在循环开头集中搬一次，避免散点和重复。当前注释里还有大段"# 关键：把 joint 也送到 GPU"这种调试痕迹，跑通后可以清理。

#### (7) YAML 兼容新版 PyYAML
```python
# 原版
default_arg = yaml.load(f)
# Windows 版
default_arg = yaml.load(f, Loader=yaml.FullLoader)
```
**原因**：新版 PyYAML 调用 `yaml.load` 不传 `Loader` 会 `YAMLLoadWarning` 或直接报错。修复后兼容 PyYAML 5.1+。

---

### 2. `model/BlockGCN.py` —— `num_person` 通用化 + 设备修复

#### (1) `TopoTrans` 增加 `num_person` 参数
```python
# 原版
class TopoTrans(nn.Module):
    def __init__(self, out_dim):
        ...
        # for ntu, two people at the same frame
        x = x.repeat(2,1)
        # for ucla, one person only
        # x = x

# Windows 版
class TopoTrans(nn.Module):
    def __init__(self, out_dim, num_person=2):
        ...
        self.num_person = num_person
        ...
        x = x.repeat(self.num_person, 1)
```
**原因**：原版需要"改源码切换 NTU/UCLA"——NTU 是双人 (`repeat(2,1)`)、UCLA 是单人 (`x=x`)。Windows 版改为通过 `num_person` 参数自动切换，与 yaml 中 `model_args.num_person` 联动。

#### (2) 模型 `__init__` 把 `num_person` 透传给所有 `TopoTrans`
```python
self.t0 = TopoTrans(out_dim=128, num_person=num_person)
self.t1 = TopoTrans(out_dim=128, num_person=num_person)
...
self.t9 = TopoTrans(out_dim=256, num_person=num_person)
```
**原因**：和 (1) 配套，让顶层 `Model(num_person=...)` 能贯穿到所有拓扑变换层。

#### (3) 设备一致性修复
```python
# 原版
x = make_tensor(x)
# Windows 版
device = x.device
...
x = make_tensor(x).to(device)
```
**原因**：`make_tensor` 默认在 CPU 上生成张量，模型在 GPU 上时会触发 device mismatch。改成跟随输入 `x` 的 device。

> 💬 **留言（优化点）**：`t0..t9` 这 10 层完全可以用 `nn.ModuleList` 写法收敛成一行，参数也只需在 init 里一次性传入；当前重复 10 行属于原版遗留，重构后更易维护，但不影响功能。

---

### 3. `torchlight/torchlight/util.py` —— PaviLogger 容错

```python
# 原版
from torchpack.runner.hooks import PaviLogger
# Windows 版
try:
    from torchpack.runner.hooks import PaviLogger
except ImportError:
    PaviLogger = None
```
**原因**：`torchpack` 是内部依赖，pip 安装容易失败 / 缺失。容错导入后即使没装也不影响整体启动。

> 💬 **留言（优化点）**：当前只挡住了导入异常，如果代码里真正调用 `PaviLogger(...)`，会在运行时 `TypeError: 'NoneType' object is not callable`。建议在使用处加判断或彻底剥离 Pavi 相关逻辑。

---

### 4. `torchlight/__init__.py` —— 新增顶层导出

```python
from .torchlight.util import IO, str2bool, str2dict, DictAction, import_class
from .torchlight.gpu import visible_gpu, occupy_gpu, ngpu
```
**原因**：原版需要 `from torchlight.torchlight.util import ...`（路径多一级），Windows 版补充顶层 `__init__.py` 后，可直接 `from torchlight import IO`，调用更整齐。

---

### 5. `train.sh` / `evaluate.sh` —— 单卡 + UCLA 示例

```bash
# 原版（多卡）
python main.py --config ... --device 0 1

# Windows 版（单卡）
python main.py --config ... --device 0
```
另外 `evaluate.sh` 追加了 UCLA 测试的命令示例（加载 `runs-65-5135.pt` 等）。

**原因**：本地大概率只有单卡，沿用 `0 1` 会因找不到第 2 张卡崩溃。脚本里同时保留了 NTU 训练命令的注释，方便复现。

> 💬 **留言（优化点）**：`train.sh` 里所有命令同时未注释会被依次执行，建议把当前要跑的留下，其它注释掉；或者改成接受参数 `bash train.sh ntu60_csub` 这种形式。

---

### 6. `config/ucla/default.yaml` —— 新增 UCLA 配置

新增了 NW-UCLA 的训练/测试配置：`num_class: 10`，`num_point: 20`，`num_person: 1`。

> ⚠️ **注意**：这里的 `model:` 字段写的是 `model.ctrgcn.Model`，与 `train.sh` 里实际用的 `--model model.BlockGCN.Model` 不一致。命令行参数会覆盖 yaml，所以能跑通，但 yaml 自身存在歧义，建议改成 `model.BlockGCN.Model` 以免误读。

---

### 7. `requirements.txt` —— 新增依赖清单

绝大多数行被注释，实际生效的是末尾 `tensorboard`、`tqdm`、`PyYAML`（被注释）、`scipy`、`torch==1.1.0` 等几行。

> 💬 **留言（优化点）**：
> 1. `torch==1.1.0` 太老，配合 CUDA 11/12 跑不起来；既然已经跑通，建议把当前真实环境 `pip freeze` 一份覆盖掉这个文件。
> 2. 大量注释行没价值，可以删除，只留实际需要的包。

---

### 8. `main copy.py` —— 中间备份

与最终 `main.py` 只差几行（少了 `warnings`、少了一段 `joint.to(...)`，注释和正式调用顺序不同）。属于开发过程产物。

> 💬 **留言（优化点）**：可以直接删掉，避免误用。

---

## 三、整体可优化方向汇总

按收益从高到低排序：

1. **`self.model.num_class` / `self.model.module.num_class` 全部换成 `self.unwrap_model().num_class`**
   - 当前单卡能跑，但一旦改 DataParallel 又会全线崩。一次性根治。

2. **统一张量搬运策略**
   - 在 Feeder 或循环入口集中 `.to(device)`，删掉中后段散落的 `joint.to(...)`，避免重复搬运 + 调试痕迹（中文注释）。

3. **`num_worker` / `prefetch_factor` 走配置**
   - 写进 yaml 而不是改 `argparse default`；在 Windows 上调成 `2~4` 通常能拉一倍吞吐。
   - 同时确认入口加了 `if __name__ == '__main__':`（Windows spawn 必备）。

4. **`config/ucla/default.yaml` 里 `model:` 改成 `model.BlockGCN.Model`**
   - 与 train.sh 保持一致，消除歧义。

5. **`requirements.txt` 重写成当前真实环境**
   - `pip freeze > requirements.txt`，删掉所有注释行。

6. **`model/BlockGCN.py` 里 `t0..t9` 收敛成 `nn.ModuleList`**
   - 纯重构，不影响精度。

7. **删除 `main copy.py`、清理 `__pycache__`/`work_dir` 中的临时产物**
   - 可加 `.gitignore`：`__pycache__/`、`work_dir/`、`*.pyc`、`data/`、`*.npz`。

8. **`PaviLogger=None` 处需要使用点的真实兜底**
   - 找到所有 `PaviLogger(...)` 调用点，按 `if PaviLogger is None: skip` 处理。

9. **`train.sh` 改成"按需启用"**
   - 同时取消注释多条命令会顺序执行，容易跑错任务。
