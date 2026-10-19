import pickle
import csv
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIGMA = 4.0
SEEDS = list(range(8))

def load_episodes(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def sample_episodes(pool, n, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]


def batch_stats(episodes):
    lengths = np.array([len(ep) for ep in episodes])
    return {"avg_len": float(lengths.mean()), "n_trans": int(lengths.sum())}


def episodes_to_arrays(episodes):
    states, actions = [], []
    for ep in episodes:
        for obs, act in ep:
            states.append(obs)
            actions.append(act)
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)


class MLP(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_bc(states, actions, seed, epochs=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
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


def evaluate(model, env, n_episodes=30, seed_start=1000):
    rewards = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed_start + i)
        total, done = 0.0, False
        while not done:
            x = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
            action = int(torch.argmax(logits, dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


expert = load_episodes("demos_expert.pkl")
noisy = load_episodes(f"demos_noisy_{SIGMA}.pkl")

groups = [
    ("10 perfect",        expert, 10),
    (f"10 noisy s{SIGMA}",  noisy, 10),
    (f"100 noisy s{SIGMA}", noisy, 100),
]
GROUP_COLORS = ["tab:green", "tab:red", "tab:orange"]

env = gym.make("CartPole-v1")
print(f"=== 三组对照（CartPole + 高斯噪声 sigma={SIGMA}）===")
results = []
points = []
for gi, (name, pool, n) in enumerate(groups):
    per_reward, per_len, per_trans = [], [], []
    for seed in SEEDS:
        eps = sample_episodes(pool, n, seed)
        st = batch_stats(eps)
        per_len.append(st["avg_len"])
        per_trans.append(st["n_trans"])
        states, actions = episodes_to_arrays(eps)
        model = train_bc(states, actions, seed)
        m, _ = evaluate(model, env)
        per_reward.append(m)
        points.append((st["avg_len"], st["n_trans"], m, GROUP_COLORS[gi]))
    mr, sr = float(np.mean(per_reward)), float(np.std(per_reward))
    ml = float(np.mean(per_len))
    mt = float(np.mean(per_trans))
    results.append((name, mr, sr, ml, mt))
    print(f"{name:16s} | 示范平均长度={ml:6.1f} | 总样本数={mt:6.0f} | "
          f"BC reward = {mr:6.1f} +/- {sr:5.1f}")

lens = np.array([p[0] for p in points])
trans = np.array([p[1] for p in points])
rews = np.array([p[2] for p in points])
cols = [p[3] for p in points]
r_len = np.corrcoef(lens, rews)[0, 1]
r_trans = np.corrcoef(trans, rews)[0, 1]
print(f"\n[仅靠数据，不训练] 示范平均长度 vs BC奖励 相关系数 = {r_len:.3f}")
print(f"[仅靠数据，不训练] 总样本数   vs BC奖励 相关系数 = {r_trans:.3f}")

with open("q2_cartpole_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "mean_reward", "std", "avg_demo_length", "n_transitions"])
    for name, mr, sr, ml, mt in results:
        w.writerow([name, mr, sr, ml, mt])

x = np.arange(len(results))
plt.figure(figsize=(8, 5))
plt.bar(x, [r[1] for r in results], yerr=[r[2] for r in results], capsize=4, color=GROUP_COLORS)
plt.xticks(x, [r[0] for r in results])
plt.ylabel("BC mean reward")
plt.ylim(0, 550)
plt.title("Q2 (CartPole, Gaussian noise)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("q2_cartpole_bar.png", dpi=150)

plt.figure(figsize=(7, 4.5))
plt.scatter(lens, rews, c=cols, alpha=0.7)
plt.xlabel("Data-only quality: mean demo length")
plt.ylabel("BC policy mean reward")
plt.title("CartPole + Gaussian noise")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("q2_cartpole_scatter.png", dpi=150)

print("\nSaved q2_cartpole_results.csv, q2_cartpole_bar.png, q2_cartpole_scatter.png")