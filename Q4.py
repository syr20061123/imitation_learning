import csv
import pickle
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist
from stable_baselines3 import PPO
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "q4_outputs"
N_DEMOS, SEEDS = 5, range(8)
STEPS, WINDOW = 200, 20
HORIZONS = [20, 50, 100, 150, 200]
EPOCHS, BATCH_SIZE, LR = 100, 256, 1e-3
K, MMD_SAMPLES = 5, 512
TEST_SEEDS = range(1000, 1100)
CALIBRATION_SEEDS = range(2000, 2050)
SCALE = np.array([1.0, 1.0, 8.0])  # cos(theta), sin(theta), angular_velocity / 8


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_demos():
    with (ROOT / "demos_pendulum_perfect.pkl").open("rb") as f:
        data = pickle.load(f)
    episodes = data["episodes"] if isinstance(data, dict) else data
    if len(episodes) < N_DEMOS or any(len(ep) != STEPS for ep in episodes):
        raise ValueError("需要至少 5 条完整的 200 步 Pendulum 示范。")
    states = np.array([[s for s, _ in ep] for ep in episodes], dtype=np.float32)
    actions = np.array([[a for _, a in ep] for ep in episodes], dtype=np.float32)
    actions = actions.reshape(len(episodes), STEPS, 1)
    if states.shape != (len(episodes), STEPS, 3):
        raise ValueError("示范状态应为 [cos(theta), sin(theta), angular_velocity]。")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("示范中存在非有限数值。")
    # 原 Q2 采集脚本使用 reset seeds 0..len(episodes)-1。
    if (min(TEST_SEEDS) < len(episodes) or min(CALIBRATION_SEEDS) < len(episodes)
            or set(TEST_SEEDS) & set(CALIBRATION_SEEDS)):
        raise ValueError("示范、校准和测试的初始状态种子必须分离。")
    return states, actions


def train_bc(states, actions, seed):
    torch.manual_seed(seed)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    x = torch.from_numpy(states.reshape(-1, 3).copy())
    y = torch.from_numpy(actions.reshape(-1, 1).copy())
    for _ in range(EPOCHS):
        order = torch.randperm(len(x))
        for start in range(0, len(x), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            loss = nn.functional.mse_loss(model(x[idx]), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def bc_actions(model, states):
    with torch.no_grad():
        actions = model(torch.as_tensor(states, dtype=torch.float32)).numpy()
    return np.clip(actions, -2.0, 2.0)


def rollout(action_fn, seeds):
    """记录动作执行前的状态，与示范格式一致；提前终止时不补零或重置。"""
    envs = [gym.make("Pendulum-v1") for _ in seeds]
    states = np.empty((len(seeds), STEPS, 3), dtype=np.float32)
    try:
        obs = np.stack([env.reset(seed=int(seed))[0] for env, seed in zip(envs, seeds)])
        for t in range(STEPS):
            states[:, t] = obs
            actions = np.asarray(action_fn(obs), dtype=np.float32).reshape(len(envs), 1)
            if not np.isfinite(actions).all():
                raise ValueError("策略输出了非有限动作。")
            actions = np.clip(actions, -2.0, 2.0)
            next_obs = []
            for env, action in zip(envs, actions):
                state, _, terminated, truncated, _ = env.step(action)
                if (terminated or truncated) and t < STEPS - 1:
                    raise RuntimeError("轨迹提前结束，不能纳入固定 Horizon 的比较。")
                next_obs.append(state)
            obs = np.stack(next_obs)
    finally:
        for env in envs:
            env.close()
    return states


def bandwidths_from(calibration):
    """只用独立校准数据确定一次局部/较宽尺度，所有种子和 Horizon 共用。"""
    points = (calibration / SCALE).reshape(-1, 3)
    rng = np.random.default_rng(31001)
    points = points[rng.choice(len(points), min(1000, len(points)), replace=False)]
    distances = pdist(points)
    positive = distances[distances > 1e-12]
    if not len(positive):
        raise ValueError("校准状态没有变化，无法确定 MMD 带宽。")
    local, broad = np.quantile(positive, [0.5, 0.9])
    return np.unique(np.r_[local * np.array([0.5, 1, 2]),
                           broad * np.array([0.5, 1, 2])])


def mmd2(x, y, bandwidths):
    """含对角项的经验 MMD²；有限样本下有偏，不作为 iid 显著性检验。"""
    def kernel_mean(a, b):
        d2 = cdist(a, b, metric="sqeuclidean")
        return np.mean([np.exp(-d2 / (2 * width ** 2)).mean() for width in bandwidths])
    return max(0.0, float(kernel_mean(x, x) + kernel_mean(y, y)
                          - 2 * kernel_mean(x, y)))


def measure(kind, start, end, seed, train, learner, expert, distances, bandwidths):
    # MMD 比较相同时间区间；BC/专家共用参考样本和配对的测试抽样位置。
    x, y, z = [(s[:, start:end] / SCALE).reshape(-1, 3) for s in (train, learner, expert)]
    rng = np.random.default_rng(np.random.SeedSequence([seed, N_DEMOS, start, end, 41001]))
    ref_idx = rng.choice(len(x), min(MMD_SAMPLES, len(x)), replace=False)
    test_idx = rng.choice(len(y), min(MMD_SAMPLES, len(y)), replace=False)
    bc_mmd = mmd2(x[ref_idx], y[test_idx], bandwidths)
    expert_mmd = mmd2(x[ref_idx], z[test_idx], bandwidths)
    bc_knn, expert_knn = [float(d[:, start:end].mean()) for d in distances]
    return dict(kind=kind, seed=seed, start=start + 1, end=end,
                knn_bc=bc_knn, knn_expert=expert_knn, knn_excess=bc_knn - expert_knn,
                mmd2_bc=bc_mmd, mmd2_expert=expert_mmd, mmd2_excess=bc_mmd - expert_mmd)


def run_experiment(verbose=True):
    torch.set_num_threads(1)
    states, actions = load_demos()
    expert_model = PPO.load(str(ROOT / "ppo_pendulum.zip"), device="cpu")
    expert_action = lambda s: expert_model.predict(s, deterministic=True)[0]
    calibration = rollout(expert_action, CALIBRATION_SEEDS)
    expert = rollout(expert_action, TEST_SEEDS)
    bandwidths = bandwidths_from(calibration)
    rows = []
    for seed in SEEDS:
        # 固定数量为 5；各种子重新抽取完整轨迹，波动含数据抽样与训练随机性。
        indices = np.random.default_rng(seed).permutation(len(states))[:N_DEMOS]
        train = states[indices]
        model = train_bc(train, actions[indices], seed)
        learner = rollout(lambda s: bc_actions(model, s), TEST_SEEDS)
        if not np.array_equal(learner[:, 0], expert[:, 0]):
            raise AssertionError("BC 和专家的测试初始状态必须相同。")
        tree = cKDTree((train / SCALE).reshape(-1, 3))
        # 最近邻始终相对该模型全部训练状态，参考集不随 Horizon 改变。
        distances = [tree.query((s / SCALE).reshape(-1, 3), k=K)[0].mean(axis=1)
                     .reshape(len(TEST_SEEDS), STEPS) for s in (learner, expert)]
        intervals = [("window", t, min(t + WINDOW, STEPS)) for t in range(0, STEPS, WINDOW)]
        intervals += [("horizon", 0, t) for t in HORIZONS]
        for kind, start, end in intervals:
            rows.append(measure(kind, start, end, seed, train, learner, expert, distances, bandwidths))
        if verbose:
            print(f"seed {seed} 完成", flush=True)
    return rows


def mean_sd(rows, kind, end, metric):
    values = [r[metric] for r in rows if r["kind"] == kind and r["end"] == end]
    return np.mean(values), np.std(values, ddof=1)


def plot_results(rows, kind, path):
    ends = sorted({r["end"] for r in rows if r["kind"] == kind})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, ylabel in zip(axes, ["knn", "mmd2"], ["Mean kNN distance", "Empirical MMD squared"]):
        for policy, color, style in [("bc", "tab:blue", "-"), ("expert", "tab:orange", "--")]:
            means, sds = np.array([mean_sd(rows, kind, t, f"{metric}_{policy}") for t in ends]).T
            ax.plot(ends, means, style, color=color, label="BC" if policy == "bc" else "Expert")
            ax.fill_between(ends, means - sds, means + sds, color=color, alpha=0.12)
        ax.set(xlabel=f"End of {WINDOW}-step window" if kind == "window" else "Horizon T",
               ylabel=ylabel, ylim=(0, None))
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{N_DEMOS} demonstrations | {len(SEEDS)} seeds | mean +/- 1 SD", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    rows = run_experiment()
    OUT.mkdir(exist_ok=True)
    # 仅保存两图一表；同名输出更新，不生成报告、权重或重复的中间文件。
    with (OUT / "q4_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(rows, "window", OUT / "q4_state_gap.png")
    plot_results(rows, "horizon", OUT / "q4_horizon.png")
    print("\nT    kNN(BC-专家)均值±SD    MMD²(BC-专家)均值±SD")
    for t in HORIZONS:
        a, b = mean_sd(rows, "horizon", t, "knn_excess")
        c, d = mean_sd(rows, "horizon", t, "mmd2_excess")
        print(f"{t:3d}  {a:.4f} ± {b:.4f}       {c:.4f} ± {d:.4f}")


if __name__ == "__main__":
    main()
