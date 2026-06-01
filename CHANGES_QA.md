# CHANGES.md 改动答疑

针对 `CHANGES.md` 提到的 Windows 适配改动,逐条回答疑问。

---

## 一、`main.py` 相关

### 1. `resource` 模块在 Windows 上不支持,没有它有什么影响?

**结论**:确实只在 Unix 系(Linux/macOS)有,Windows 没有这个包;但在本项目里**没有任何实际功能影响**。

- `resource` 是 Python 标准库,但**仅在 POSIX 系统**(Linux/macOS)上提供。Windows 的 Python 不带这个模块,`import resource` 会直接 `ImportError` 让脚本根本起不来。
- 原版用它做了**一件事**:

  ```python
  rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
  ```

  含义是把当前进程"**可同时打开的文件描述符上限**"(`RLIMIT_NOFILE`)抬到 2048。原因是 DataLoader 多 worker + 大量 `.npz` 切片同时打开时,Linux 默认 1024 容易撞到 `Too many open files` 报错。

- Windows 根本没有"文件描述符 ulimit"这个概念(它用 handle,而且默认上限非常大,16384+),所以**不需要也无法设置**。
- Windows 版用 `try/except ImportError: resource = None` 把它变成可选,完美绕开。**对训练精度、速度、收敛都没有任何影响**,纯启动兼容性。

---

### 2. `prefetch_factor` 是干嘛的?和 `num_worker` 什么关系?

**结论**:`prefetch_factor` 控制**每个 worker 预先准备多少个 batch**,只有在 `num_worker > 0`(多进程加载)时才有意义。

- PyTorch `DataLoader` 的工作模式:
  - `num_worker = 0`:**主进程**自己加载数据,GPU 计算时 CPU 在闲着,加载和训练**串行**,慢。
  - `num_worker > 0`:开 N 个**子进程**后台加载,主进程从队列里取已经准备好的 batch,加载和训练**并行**。

- `prefetch_factor` = 每个 worker 在队列里**最多缓冲多少个 batch**(默认 2)。
  - 总缓冲 batch 数 = `num_worker × prefetch_factor`。
  - 例如 `num_worker=4, prefetch_factor=2` → 队列里随时有 8 个 batch 待命,GPU 几乎不会等 I/O。
  - 设大了:占内存(每个 batch 都驻留在内存),CPU 也会忙着提前准备用不上的数据。
  - 设小了:GPU 跑得快、CPU 跟不上时会出现 GPU 等数据 → 利用率掉。

- **关键约束**:PyTorch 规定 `num_worker=0` 时**不能传 `prefetch_factor`**(没 worker 谈何预取),会直接抛 `ValueError`。
- 原版无条件传 `prefetch_factor=16`,在 Windows 上把 `num_worker` 调成 0 后就报错。Windows 版改成动态拼装:

  ```python
  loader_args = dict(num_workers=self.arg.num_worker, worker_init_fn=init_seed)
  if self.arg.num_worker > 0:
      loader_args['prefetch_factor'] = 2
  ```

  `num_worker=0` 时根本不传这个参数,两种情况都能跑。

- 调优建议:跑通后把 `num_worker` 调到 2~4、`prefetch_factor` 设 2~4 试一下,Windows 下吞吐通常能翻倍。但 Windows 必须用 `spawn` 启动子进程,**入口要加 `if __name__ == '__main__':` 守卫**,否则子进程会递归再启动整个脚本。

---

### 3. `self.model.module.num_class` 是 DataParallel 包装的吗?单卡能用吗?

**结论**:是的,`.module` 是 `DataParallel` / `DistributedDataParallel` 的标志,单卡不包装就**没有** `.module`,直接访问会 `AttributeError`。

- `nn.DataParallel(model)` 会把原模型**包一层**,把原模型放进 `.module` 属性,然后自己负责往多卡 scatter/gather:

  ```
  原模型:        model                    →  model.num_class ✓
  DataParallel: DataParallel(model)       →  model.module.num_class ✓
                                              model.num_class ✗ AttributeError
  ```

- 所以**有没有 `.module` 取决于是否被包装**,不是单纯看几张卡:
  - 单卡 + 不包装(本项目当前情况)→ 用 `self.model.num_class`
  - 多卡 `DataParallel` → 必须用 `self.model.module.num_class`
  - 你当前代码全是 `self.model.num_class`,所以单卡能跑;**一旦改回 DataParallel 就会全线 AttributeError**。

---

### 4. `unwrap_model` 函数有啥用?

**结论**:它是一个"**单卡多卡通用适配器**",一行代码搞定"有没有 `.module`"的判断。

```python
def unwrap_model(self):
    return self.model.module if hasattr(self.model, 'module') else self.model
```

- 逻辑:**如果**模型有 `.module`(被 DataParallel 包过),返回里层真模型;**否则**直接返回 `self.model`。
- 这样调用处统一写:

  ```python
  self.unwrap_model().num_class   # 单卡多卡都对
  ```

  就不用关心当前是 `DataParallel(Model)` 还是裸 `Model`。

- 但当前 Windows 版只**定义了这个函数,却没真正用**(`load_model` 之后调用处仍是 `self.model.num_class`,`unwrap_model()` 的写法被写在注释里)。所以现状是:**单卡能跑,但失去了多卡可移植性**。
- 建议把所有 `self.model.num_class` / `self.model.module.num_class` 统一换成 `self.unwrap_model().num_class`,一次性根治。

---

### 5. `yaml.load(f, Loader=yaml.FullLoader)` 为什么要加 `Loader`?是版本问题吗?

**结论**:是的,纯版本兼容问题。PyYAML 5.1 之后,`yaml.load()` 不显式传 `Loader` 会**报警告或报错**。

- 背景:`yaml.load()` 默认行为可以**反序列化任意 Python 对象**(包括执行任意代码),曾经爆出过远程命令执行的安全漏洞。
- **PyYAML ≥ 5.1** 把这个行为收紧了:
  - 不传 `Loader` → `YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated`(老版本只警告)
  - **PyYAML ≥ 6.0** → 直接 `TypeError: load() missing 1 required positional argument: 'Loader'`,**根本跑不动**。
- 显式指定 `Loader` 告诉 yaml 用哪种解析策略:
  - `yaml.FullLoader`:能解析所有 YAML 标签,但不会执行任意 Python 对象(安全)。**推荐**,本项目用的就是这个。
  - `yaml.SafeLoader`:只解析基础类型(dict/list/str/int 等),最严格。
  - `yaml.UnsafeLoader` / `yaml.Loader`:旧的不安全行为,不要用。
- 原版 `yaml.load(f)` 是 2019 年以前的写法,在新环境下必坏。Windows 版加上 `Loader=yaml.FullLoader` 兼容 PyYAML 5.1+/6.x。

---

## 二、`model/BlockGCN.py` 相关

### 6. `num_person` 适配 + `x.device` 适配,理解对吗?

**完全正确**,可以从两个角度展开:

#### (a) `num_person` 通用化:NTU(双人)/ UCLA(单人)的"动作-人"维度差异

- 骨架数据张量形状一般是 `(N, C, T, V, M)`,其中 `M = num_person`(同一时刻最多几个人)。
- **NTU-60 / NTU-120**:包含双人交互动作(握手、拥抱等),`M = 2`。
- **NW-UCLA**:全是单人动作,`M = 1`。
- 模型里的 `TopoTrans` 需要把"骨架拓扑"广播到每个人身上,原版硬编码:

  ```python
  x = x.repeat(2, 1)   # 给 NTU 用
  # x = x              # 给 UCLA 用,要手改
  ```

  切数据集就得改源码,非常脏。
- Windows 版改成参数化:

  ```python
  def __init__(self, out_dim, num_person=2):
      self.num_person = num_person
      ...
      x = x.repeat(self.num_person, 1)
  ```

  然后顶层 `Model(num_person=...)` 把这个值从 yaml `model_args.num_person` 一路透传到所有 10 个 `TopoTrans` 层。换数据集只改配置文件,不动代码。

#### (b) `x.device` 适配:解决"张量不在同一设备"的报错

- PyTorch 强制要求**参与运算的所有张量必须在同一个设备**(同一张 GPU 或都在 CPU),否则:

  ```
  RuntimeError: Expected all tensors to be on the same device,
  but found at least two devices, cuda:0 and cpu!
  ```

- 原版 `TopoTrans` 里:

  ```python
  x = make_tensor(x)   # make_tensor 默认在 CPU 上 new 一个张量
  ```

  但模型本身已经 `.cuda()` 在 GPU 上跑,这一句立刻就把 CPU 张量塞进 GPU 计算图,炸。

- Windows 版修复:

  ```python
  device = x.device         # 看输入 x 在哪个设备
  ...
  x = make_tensor(x).to(device)   # 跟随输入,无论 CPU/GPU 都对
  ```

  原理就是"**计算用的辅助张量,要主动跟到主输入所在的设备上**"。这种 device-follow 写法是 PyTorch 自定义层的标准做法。

> 顺带提一句:`main.py` 里那些 `data.to(self.output_device)`、`joint.to(self.output_device)` 也是为了堵同样的 device mismatch,只不过是在外层 train loop 里堵。理想做法是在 Feeder 或循环入口集中搬一次,而不是散落多处(`CHANGES.md` 已列为优化点 2)。

---

## 三、`torchlight/util.py` 相关

### 7. `PaviLogger` 是干嘛的?

**结论**:它是 `torchpack`(商汤内部训练框架)里的一个**远程日志上报组件**,作用类似 TensorBoard/WandB,把训练 loss/acc 实时推送到 **Pavi 平台**(商汤内部的可视化看板)。

- 用途场景:商汤内部跑大规模分布式训练时,需要在网页 dashboard 上集中看每个实验的 loss 曲线、GPU 利用率等,`PaviLogger` 就是那个上报客户端。
- 在公开的 PyPI 上**通常装不到**,或者装了也连不上内部服务(需要内网鉴权)。对外开源的 BlockGCN 仓库带着这个 import 是历史遗留。
- 对本项目而言:**完全用不到**。本地有 TensorBoard 看曲线就够了。
- 原版直接 `from torchpack.runner.hooks import PaviLogger`,在外部环境 `torchpack` 没装就直接 `ImportError` 退出。Windows 版改成:

  ```python
  try:
      from torchpack.runner.hooks import PaviLogger
  except ImportError:
      PaviLogger = None
  ```

  装不上就当它不存在,**只要后续不主动调用 `PaviLogger(...)`**,啥事没有。
- 留的尾巴:如果代码里某处**真的去调** `PaviLogger(...)`,`None(...)` 会触发 `TypeError: 'NoneType' object is not callable`。所以更稳的写法是在所有使用点加 `if PaviLogger is None: skip`,或者彻底把 Pavi 相关分支删掉。本项目当前没有真实调用点,所以现在能跑;但属于隐藏地雷。

---

## 总结速查表

| 疑问 | 一句话答案 |
|---|---|
| `resource` Windows 没有 | 真没有,但只用来抬文件描述符上限,Windows 不需要,纯启动兼容 |
| `prefetch_factor` 干嘛 | 每个 worker 预取多少个 batch,`num_worker=0` 时**禁止传** |
| `model.module.num_class` | `.module` 来自 DataParallel 包装,单卡裸模型没这个属性 |
| `unwrap_model` 干嘛 | 单卡多卡通用的"剥壳"工具,统一拿到真模型 |
| `yaml.load(Loader=...)` | PyYAML 5.1+ 强制要求,不传会 warn/报错 |
| `num_person` 改动 | NTU=2 / UCLA=1,参数化后不用改源码切数据集 |
| `x.device` 改动 | 让辅助张量跟主输入到同一设备,避免 device mismatch |
| `PaviLogger` | 商汤内部日志推送服务,公开环境装不上,容错导入忽略即可 |
