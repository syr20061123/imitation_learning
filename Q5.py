"""Q5：仅比较 K=1、5、20 的动作分块；不包含 ACT 时间集成。
用二维闭环位置伺服轨迹测量实际状态误差的累计频率和块间衔接误差。
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "q5_outputs"
SEEDS, N_DEMOS, STEPS, N_TEST = range(8), 5, 200, 100
CHUNKS = (1, 5, 20)
METHODS = tuple((k, "分块") for k in CHUNKS)
NOISE_LEVELS = (0.0, 0.0005)  # 米：无噪声 / 0.5 毫米观测噪声，所有方法共用。
RADIUS, DT, INNER_STEPS, SERVO_TAU, OMEGA = 0.10, 0.02, 5, 0.015, 0.6
INNER_ALPHA = 1 - np.exp(-DT / INNER_STEPS / SERVO_TAU)
ALPHA = 1 - np.exp(-DT / SERVO_TAU)
ANGLE = OMEGA * DT
ROTATION = np.array([[np.cos(ANGLE), -np.sin(ANGLE)], [np.sin(ANGLE), np.cos(ANGLE)]])
EPOCHS, BATCH_SIZE, LR = 200, 256, 1e-3
GROWTH_TOL = 1e-3  # 平方毫米；忽略近零变化，统计有实际幅度的误差增长步。


def initial_states(seed, count):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, count)
    radius = rng.uniform(0.8 * RADIUS, 1.2 * RADIUS, count)
    return radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def servo_step(q, target):
    # 250 Hz 一阶闭环位置伺服；高层每 20 毫秒发送绝对位置目标。
    # 各 K 的底层反馈均保持开启，不把位置目标当作开环力矩。
    q = q.copy()
    for _ in range(INNER_STEPS):
        q += INNER_ALPHA * (target - q)
    return q


def expert_action(q):
    # 确定性专家：让下一帧位置沿圆弧前进 ANGLE，补偿同一个伺服器的跟踪滞后。
    return (q @ ROTATION.T - (1 - ALPHA) * q) / ALPHA


def expert_path(q0):
    q = np.empty((len(q0), STEPS + 1, 2))
    a = np.empty((len(q0), STEPS, 2))
    q[:, 0] = q0
    for t in range(STEPS):
        a[:, t] = expert_action(q[:, t])
        q[:, t + 1] = servo_step(q[:, t], a[:, t])
    return q, a


class ChunkMLP(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        # 所有 K 使用完全相同的初始化和骨干；只监督/执行前 K 个输出头。
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2 * max(CHUNKS)))

    def forward(self, x):
        return self.net(x).reshape(-1, max(CHUNKS), 2)[:, :self.k]


def train_model(states, actions, k, seed):
    torch.manual_seed(seed)
    model = ChunkMLP(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    x = torch.tensor(states[:, :-1].reshape(-1, 2) / RADIUS, dtype=torch.float32)
    index = np.arange(STEPS)[:, None] + np.arange(k)[None, :]
    mask = torch.tensor(np.broadcast_to(index < STEPS, (N_DEMOS, STEPS, k)).copy().reshape(-1, k))
    y = torch.tensor(actions[:, np.minimum(index, STEPS - 1)] / RADIUS,
                     dtype=torch.float32).reshape(-1, k, 2)
    generator = torch.Generator().manual_seed(seed + 51001)
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            error = (model(x[idx]) - y[idx]).square().mean(dim=-1) * mask[idx]
            loss = (error.sum(dim=1) / mask[idx].sum(dim=1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def predict(model, observation):
    with torch.no_grad():
        actions = model(torch.tensor(observation / RADIUS, dtype=torch.float32)).numpy() * RADIUS
    if not np.isfinite(actions).all():
        raise ValueError("预测动作包含非有限数值。")
    return np.clip(actions, -0.25, 0.25)


def rollout(model, q0, noise):
    q = np.empty((len(q0), STEPS + 1, 2))
    actions = np.empty((len(q0), STEPS, 2))
    q[:, 0] = q0
    for t in range(STEPS):
        observed = q[:, t] + noise[:, t]
        if t % model.k == 0:
            chunk = predict(model, observed)
        actions[:, t] = chunk[:, t % model.k]
        q[:, t + 1] = servo_step(q[:, t], actions[:, t])
    return q, actions


def measure(q, reference_q, k):
    # 以真实任务位置为基准，包含之前动作造成的偏移；没有用未来真值指导实际策略。
    error = np.sum(((q - reference_q) * 1000) ** 2, axis=-1)
    mean_error = error.mean(axis=0)
    # 边界衔接用状态误差增量衡量，并扣除相邻块内步的平均增量。
    increment = np.diff(error, axis=1)
    seam = np.full(STEPS, np.nan)
    if k > 1:
        boundary = np.arange(k, STEPS, k)
        seam[boundary] = increment[:, boundary].mean(axis=0) - (
            increment[:, boundary - 1].mean(axis=0) + increment[:, boundary + 1].mean(axis=0)
        ) / 2
    values = dict(
        state_error=mean_error[1:], cumulative_error=np.cumsum(mean_error[1:]),
        growth_rate=(increment > GROWTH_TOL).mean(axis=0),
        junction_state_error_growth=seam,
    )
    return [{key: (float(v[t]) if np.isfinite(v[t]) else None) for key, v in values.items()}
            for t in range(STEPS)]


def run_experiment(seeds=SEEDS, test_seed=20000, n_test=N_TEST, verbose=True):
    torch.set_num_threads(1)
    q0 = initial_states(test_seed, n_test)
    reference_q, _ = expert_path(q0)
    rows = []
    for seed in seeds:
        demo_q, demo_a = expert_path(initial_states(1000 + seed, N_DEMOS))
        rng = np.random.default_rng(30000 + seed)
        common_noise = rng.normal(size=(n_test, STEPS, 2))
        for k in CHUNKS:
            model = train_model(demo_q, demo_a, k, seed)
            for noise_sd in NOISE_LEVELS:
                noise = common_noise * noise_sd
                for _, mode in [m for m in METHODS if m[0] == k]:
                    q, actions = rollout(model, q0, noise)
                    records = measure(q, reference_q, k)
                    for t, record in enumerate(records):
                        rows.append(dict(seed=seed, chunk_size=k, mode=mode,
                                         noise_mm=noise_sd * 1000, step=t + 1,
                                         **record))
        if verbose:
            print(f"种子 {seed} 完成", flush=True)
    return rows


def series(rows, k, mode, noise_mm, metric):
    selected = sorted((r for r in rows if (r["chunk_size"], r["mode"], r["noise_mm"]) ==
                       (k, mode, noise_mm)), key=lambda r: (r["seed"], r["step"]))
    return np.array([r[metric] for r in selected], dtype=float).reshape(-1, STEPS)


def mean_sd(rows, k, mode, noise, metric, final=False):
    a = series(rows, k, mode, noise, metric)
    a = a[:, -1] if final else np.nanmean(a, axis=1)
    return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0


def plot_results(rows, path):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    noise = 0.5
    colors = {1: "tab:blue", 5: "tab:orange", 20: "tab:green"}
    for k, mode in METHODS:
        values = series(rows, k, mode, noise, "cumulative_error")
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(STEPS)
        axes[0].plot(np.arange(1, STEPS + 1), mean, color=colors[k],
                     label=f"分块 K={k}")
        axes[0].fill_between(np.arange(1, STEPS + 1), mean - sd, mean + sd,
                             color=colors[k], alpha=0.08)
    axes[0].set(xlabel="执行步数", ylabel="累计轨迹平方误差（平方毫米·步）",
                title="观测噪声 0.5 毫米")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8)

    labels = [f"K={k}" for k, _ in METHODS]
    for noise, color in [(0.0, "tab:blue"), (0.5, "tab:orange")]:
        means, sds = np.array([mean_sd(rows, k, mode, noise, "growth_rate")
                               for k, mode in METHODS]).T
        axes[1].errorbar(np.arange(len(METHODS)), means * 100, yerr=sds * 100,
                         fmt="o-", capsize=3, color=color,
                         label=f"观测噪声 {noise:g} 毫米")
    axes[1].set_xticks(np.arange(len(METHODS)), labels, fontsize=8)
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].set(xlabel="执行方式", ylabel="误差增长步占比（%）", ylim=(0, 100))
    axes[1].legend(frameon=False, fontsize=8)

    seam_methods = [method for method in METHODS if method[0] > 1 and method[1] == "分块"]
    seam_labels = [f"K={k}" for k, _ in seam_methods]
    for noise, color in [(0.0, "tab:blue"), (0.5, "tab:orange")]:
        means, sds = np.array([mean_sd(rows, k, mode, noise,
                                      "junction_state_error_growth")
                               for k, mode in seam_methods]).T
        axes[2].errorbar(np.arange(len(seam_methods)), means, yerr=sds,
                         fmt="o-", capsize=3, color=color,
                         label=f"观测噪声 {noise:g} 毫米")
    axes[2].set_xticks(np.arange(len(seam_methods)), seam_labels)
    axes[2].set(xlabel="分块长度", ylabel="边界额外轨迹误差增量（平方毫米）",
                title="块边界与相邻块内步比较")
    axes[2].axhline(0, color="gray", linewidth=0.7)
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"位置伺服轨迹跟踪｜5 条示范｜{len(set(r['seed'] for r in rows))} 个配对种子｜均值 ± 1 个标准差")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    rows = run_experiment(verbose=False)
    OUT.mkdir(exist_ok=True)
    with (OUT / "q5_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(rows, OUT / "q5_comparison.png")
    print("\n0.5 毫米观测噪声：分块长度、误差增长步占比、200 步累计轨迹误差")
    for k, mode in METHODS:
        growth, growth_sd = mean_sd(rows, k, mode, 0.5, "growth_rate")
        error, error_sd = mean_sd(rows, k, mode, 0.5, "cumulative_error", final=True)
        print(f"K={k}：{growth:.1%} ± {growth_sd:.1%}；{error:.1f} ± {error_sd:.1f}")
    print("块边界额外轨迹误差增量（平方毫米）：")
    for k, mode in METHODS:
        if k > 1 and mode == "分块":
            mean, sd = mean_sd(rows, k, mode, 0.5, "junction_state_error_growth")
            print(f"K={k}：{mean:.4f} ± {sd:.4f}")


if __name__ == "__main__":
    main()
