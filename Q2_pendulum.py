import os
import pickle
import csv
import numpy as np
np.float_ = np.float64  # 防御性补丁，无害
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIGMA = 2.0
EXPERT_PATH = "ppo_pendulum.zip"
PERFECT_FILE = "demos_pendulum_perfect.pkl"
NOISY_FILE = f"demos_pendulum_noisy_{SIGMA}.pkl"
TRAIN_STEPS = 1_000_000
SEEDS = list(range(8))


# ============ 1. 训练/加载专家 ============
def get_expert():
    env = gym.make("Pendulum-v1")
    if os.path.exists(EXPERT_PATH):
        return PPO.load(EXPERT_PATH, env=env)
    model = PPO("MlpPolicy", env, seed=1, verbose=0)
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(EXPERT_PATH)
    mr, sr = evaluate_policy(model, env, n_eval_episodes=30)
    print(f"[expert] reward = {mr:.1f} +/- {sr:.1f}  (越小/越接近0越好)")
    return model


# ============ 2. 采集示范（含真实累计奖励） ============
def collect(expert, env, sigma, n=200, seed0=0):
    rng = np.random.RandomState(seed0)
    eps, rets = [], []
    for i in range(n):
        obs, info = env.reset(seed=i)
        ep, tot = [], 0.0
        done = False
        while not done:
            det = float(expert.predict(obs, deterministic=True)[0].item())
            a = det
            if sigma > 0.0:
                a = float(np.clip(det + sigma * rng.randn(), -2.0, 2.0))
            ep.append((obs.copy(), a))
            obs, r, term, trunc, info = env.step(np.array([a], np.float32))
            tot += r
            done = term or trunc
        eps.append(ep)
        rets.append(tot)
    return {"episodes": eps, "returns": rets}


def load_or_collect(path, expert, env, sigma):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    data = collect(expert, env, sigma)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"collected & saved {path}: {len(data['episodes'])} eps, "
          f"avg return = {np.mean(data['returns']):.1f}")
    return data


def sample_demos(data, n, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(data["episodes"]), size=n, replace=False)
    eps = [data["episodes"][i] for i in idx]
    rets = [data["returns"][i] for i in idx]
    return eps, rets


def batch_stats(eps, rets):
    return {
        "avg_return": float(np.mean(rets)),
        "n_trans": int(sum(len(ep) for ep in eps)),
    }


# ============ 3. BC 模型（回归） ============
class MLP(nn.Module):
    def __init__(self, obs_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def episodes_to_arrays(eps):
    s, a = [], []
    for ep in eps:
        for obs, act in ep:
            s.append(obs)
            a.append(act)
    return np.array(s, np.float32), np.array(a, np.float32).reshape(-1, 1)


def train_bc(states, actions, seed, epochs=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X = torch.from_numpy(states)
    Y = torch.from_numpy(actions)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            loss = loss_fn(model(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def evaluate_bc(model, env, n_episodes=30, seed_start=1000):
    rewards = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed_start + i)
        total, done = 0.0, False
        while not done:
            x = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                a = float(model(x).item())
            a = float(np.clip(a, -2.0, 2.0))
            obs, r, term, trunc, info = env.step(np.array([a], np.float32))
            total += r
            done = term or trunc
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


# ============ 4. 主流程 ============
env = gym.make("Pendulum-v1")
expert = get_expert()
perfect = load_or_collect(PERFECT_FILE, expert, env, sigma=0.0)
noisy = load_or_collect(NOISY_FILE, expert, env, sigma=SIGMA)

groups = [
    ("10 perfect",        perfect, 10),
    (f"10 noisy s{SIGMA}", noisy,   10),
    (f"100 noisy s{SIGMA}", noisy,  100),
]
GROUP_COLORS = ["tab:green", "tab:red", "tab:orange"]

print("\n=== 三组对照：数据质量 vs 训练后的 BC 表现 ===")
results = []
points = []
for gi, (name, data, n) in enumerate(groups):
    per_reward, per_ret, per_trans = [], [], []
    for seed in SEEDS:
        eps, rets = sample_demos(data, n, seed)
        st = batch_stats(eps, rets)
        per_ret.append(st["avg_return"])
        per_trans.append(st["n_trans"])
        states, actions = episodes_to_arrays(eps)
        model = train_bc(states, actions, seed)
        m, _ = evaluate_bc(model, env)
        per_reward.append(m)
        points.append((st["avg_return"], st["n_trans"], m, GROUP_COLORS[gi]))
    mr, sr = float(np.mean(per_reward)), float(np.std(per_reward))
    mret = float(np.mean(per_ret))
    mtr = float(np.mean(per_trans))
    results.append((name, mr, sr, mret, mtr))
    print(f"{name:16s} | 示范平均奖励={mret:8.1f} | 总样本数={mtr:6.0f} | "
          f"BC reward = {mr:8.1f} +/- {sr:6.1f}")

rets_arr = np.array([p[0] for p in points])
trans_arr = np.array([p[1] for p in points])
rew_arr = np.array([p[2] for p in points])
cols = [p[3] for p in points]
r_ret = np.corrcoef(rets_arr, rew_arr)[0, 1]
r_trans = np.corrcoef(trans_arr, rew_arr)[0, 1]
print(f"\n[仅靠数据，不训练] 示范平均奖励 vs BC奖励 相关系数 = {r_ret:.3f}")
print(f"[仅靠数据，不训练] 总样本数   vs BC奖励 相关系数 = {r_trans:.3f}")

with open("q2_pendulum_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "mean_reward", "std", "avg_demo_return", "n_transitions"])
    for name, mr, sr, mret, mtr in results:
        w.writerow([name, mr, sr, mret, mtr])

# 图1：三组柱状图
x = np.arange(len(results))
plt.figure(figsize=(8, 5))
plt.bar(x, [r[1] for r in results], yerr=[r[2] for r in results], capsize=4, color=GROUP_COLORS)
plt.xticks(x, [r[0] for r in results])
plt.ylabel("BC mean reward")
plt.title("Q2 (Pendulum-v1): 10 perfect vs 10 noisy vs 100 noisy")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("q2_pendulum_bar.png", dpi=150)

# 图2：散点（横轴=示范平均奖励，颜色=组）
plt.figure(figsize=(7, 4.5))
plt.scatter(rets_arr, rew_arr, c=cols, alpha=0.7)
plt.xlabel("Data-only quality: mean demo return")
plt.ylabel("BC policy mean reward")
plt.title("mean return predicts quality, but quantity also matters")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("q2_pendulum_scatter.png", dpi=150)

print("\nSaved q2_pendulum_results.csv, q2_pendulum_bar.png, q2_pendulum_scatter.png")