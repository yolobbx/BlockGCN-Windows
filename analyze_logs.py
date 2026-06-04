import os
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

UCLA_DIR = r"F:\code\BlockGCN\work_dir\ucla\140_epochs_j"
NTU_DIR = r"F:\code\BlockGCN\work_dir\ntu60\csub\140_epochs_j"
OUT_DIR = r"F:\code\BlockGCN"


def load_events(d):
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        evs = ea.Scalars(tag)
        out[tag] = [(e.step, e.value) for e in evs]
    return out


def parse_log(path):
    train_acc, train_loss = [], []
    eval_acc, eval_loss = [], []
    cur_train_epoch = None
    cur_eval_epoch = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"Training epoch:\s*(\d+)", line)
            if m:
                cur_train_epoch = int(m.group(1))
                continue
            m = re.search(r"Eval epoch:\s*(\d+)", line)
            if m:
                cur_eval_epoch = int(m.group(1))
                continue
            m = re.search(r"Mean training loss:\s*(\d+(?:\.\d+)?).*Mean training acc:\s*(\d+(?:\.\d+)?)%", line)
            if m and cur_train_epoch is not None:
                train_loss.append((cur_train_epoch, float(m.group(1))))
                train_acc.append((cur_train_epoch, float(m.group(2))))
                continue
            m = re.search(r"Mean test loss of \d+ batches:\s*(\d+(?:\.\d+)?)", line)
            if m and cur_eval_epoch is not None:
                eval_loss.append((cur_eval_epoch, float(m.group(1))))
                continue
            m = re.search(r"Top1:\s*(\d+(?:\.\d+)?)%", line)
            if m and cur_eval_epoch is not None:
                eval_acc.append((cur_eval_epoch, float(m.group(1))))
                continue
    # only keep last occurrence per epoch (in case of restarts)
    def dedup(xs):
        d = {}
        for k, v in xs:
            d[k] = v
        return sorted(d.items())
    return dedup(train_acc), dedup(train_loss), dedup(eval_acc), dedup(eval_loss)


def best_eval(eval_acc):
    if not eval_acc:
        return None, None
    e, a = max(eval_acc, key=lambda x: x[1])
    return e, a


def main():
    print("=== UCLA tb tags ===")
    ucla_train = load_events(os.path.join(UCLA_DIR, "runs", "train"))
    ucla_val = load_events(os.path.join(UCLA_DIR, "runs", "val"))
    print("train tags:", list(ucla_train.keys()))
    print("val tags:", list(ucla_val.keys()))
    for t, v in ucla_train.items():
        print(f"  train/{t}: {len(v)} pts")
    for t, v in ucla_val.items():
        print(f"  val/{t}: {len(v)} pts")

    print("=== NTU tb tags ===")
    ntu_train = load_events(os.path.join(NTU_DIR, "runs", "train"))
    ntu_val = load_events(os.path.join(NTU_DIR, "runs", "val"))
    print("train tags:", list(ntu_train.keys()))
    print("val tags:", list(ntu_val.keys()))

    # Parse logs
    u_ta, u_tl, u_ea, u_el = parse_log(os.path.join(UCLA_DIR, "log.txt"))
    n_ta, n_tl, n_ea, n_el = parse_log(os.path.join(NTU_DIR, "log.txt"))

    print("UCLA best eval:", best_eval(u_ea))
    print("NTU60 best eval:", best_eval(n_ea))
    print("UCLA last train epoch:", u_ta[-1] if u_ta else None)
    print("NTU last train epoch:", n_ta[-1] if n_ta else None)

    # Plot UCLA: train vs eval acc; train vs eval loss using log-parsed data (denser)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    if u_ta:
        x, y = zip(*u_ta)
        ax.plot(x, y, label="Train Acc", color="tab:blue")
    if u_ea:
        x, y = zip(*u_ea)
        ax.plot(x, y, label="Eval Top1 Acc", color="tab:orange")
        be, ba = best_eval(u_ea)
        ax.axvline(be, ls="--", color="red", alpha=0.5)
        ax.scatter([be], [ba], color="red", zorder=5, label=f"Best: ep{be}, {ba:.2f}%")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title("UCLA: Train vs Eval Accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    if u_tl:
        x, y = zip(*u_tl)
        ax.plot(x, y, label="Train Loss", color="tab:blue")
    if u_el:
        x, y = zip(*u_el)
        ax.plot(x, y, label="Eval Loss", color="tab:orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("UCLA: Train vs Eval Loss")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ucla_train_vs_eval.png"), dpi=120)
    plt.close()

    # Same for NTU60
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    if n_ta:
        x, y = zip(*n_ta)
        ax.plot(x, y, label="Train Acc", color="tab:blue")
    if n_ea:
        x, y = zip(*n_ea)
        ax.plot(x, y, label="Eval Top1 Acc", color="tab:orange")
        be, ba = best_eval(n_ea)
        ax.axvline(be, ls="--", color="red", alpha=0.5)
        ax.scatter([be], [ba], color="red", zorder=5, label=f"Best: ep{be}, {ba:.2f}%")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title("NTU60-CSub: Train vs Eval Accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    if n_tl:
        x, y = zip(*n_tl)
        ax.plot(x, y, label="Train Loss", color="tab:blue")
    if n_el:
        x, y = zip(*n_el)
        ax.plot(x, y, label="Eval Loss", color="tab:orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("NTU60-CSub: Train vs Eval Loss")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ntu60_train_vs_eval.png"), dpi=120)
    plt.close()

    # Also plot from tensorboard data for UCLA (as requested)
    def get(d, candidates):
        for c in candidates:
            if c in d:
                return d[c]
        return None

    tb_train_acc = get(ucla_train, ["acc", "acc/epoch", "train/acc"])
    tb_train_loss = get(ucla_train, ["loss", "loss/epoch", "train/loss"])
    tb_val_acc = get(ucla_val, ["acc", "acc/epoch", "val/acc"])
    tb_val_loss = get(ucla_val, ["loss", "loss/epoch", "val/loss"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    if tb_train_acc:
        x, y = zip(*tb_train_acc)
        ax.plot(x, y, label="Train Acc (TB)", color="tab:blue")
    if tb_val_acc:
        x, y = zip(*tb_val_acc)
        ax.plot(x, y, label="Eval Acc (TB)", color="tab:orange")
    ax.set_xlabel("Step / Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("UCLA (TensorBoard): Train vs Eval Accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    if tb_train_loss:
        x, y = zip(*tb_train_loss)
        ax.plot(x, y, label="Train Loss (TB)", color="tab:blue")
    if tb_val_loss:
        x, y = zip(*tb_val_loss)
        ax.plot(x, y, label="Eval Loss (TB)", color="tab:orange")
    ax.set_xlabel("Step / Epoch"); ax.set_ylabel("Loss"); ax.set_title("UCLA (TensorBoard): Train vs Eval Loss")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ucla_tensorboard.png"), dpi=120)
    plt.close()
    print("Saved plots to", OUT_DIR)


if __name__ == "__main__":
    main()
