# -*- coding: utf-8 -*-
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import Rectangle

SEEDS = list(range(8))
K = 2.0            # 反馈增益
TARGET_L = -0.5    # 左车道
TARGET_R = 0.5     # 右车道


class PointObstacleEnv:
    """点智能体向上匀速走、控制横向速度，中间一块障碍物，左右都能绕过。"""
    def __init__(self):
        self.dt = 0.1
        self.vy = 0.05
        self.obs_x = (-0.2, 0.2)
        self.obs_y = (0.4, 0.7)
        self.state = None

    def reset(self):
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        return self.state.copy()

    def step(self, u):
        u = float(np.clip(u, -1.0, 1.0))
        x, y = self.state
        x = x + u * self.dt
        y = y + self.vy
        self.state = np.array([x, y], dtype=np.float32)
        if abs(x) > 1.0:
            return self.state.copy(), -1.0, True
        if y >= 1.0:
            return self.state.copy(), 1.0, True
        if self.obs_x[0] <= x <= self.obs_x[1] and self.obs_y[0] <= y <= self.obs_y[1]:
            return self.state.copy(), -1.0, True
        return self.state.copy(), 0.0, False


def expert_action(s, mode):
    target = TARGET_L if mode == "left" else TARGET_R
    return float(np.clip(-K * (s[0] - target), -1.0, 1.0))


def collect(env, mode, n, noise=0.1):
    rng = np.random.RandomState(0)
    eps = []
    for _ in range(n):
        s = env.reset()
        ep = []
        done = False
        while not done:
            a = expert_action(s, mode) + rng.normal(0.0, noise)
            ep.append((s.copy(), float(a)))
            s, r, done = env.step(a)
        eps.append(ep)
    return eps


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def to_arrays(eps):
    s, a = [], []
    for ep in eps:
        for obs, act in ep:
            s.append(obs)
            a.append(act)
    return np.array(s, np.float32), np.array(a, np.float32).reshape(-1, 1)


def train_bc(states, actions, seed, epochs=200):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = MLP()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lf = nn.MSELoss()
    X = torch.from_numpy(states)
    Y = torch.from_numpy(actions)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            loss = lf(m(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


def rollout(model, env):
    s = env.reset()
    xs, ys = [s[0]], [s[1]]
    done = False
    while not done:
        x = torch.from_numpy(s).float().unsqueeze(0)
        with torch.no_grad():
            a = float(model(x).item())
        s, r, done = env.step(a)
        xs.append(s[0])
        ys.append(s[1])
    return np.array(xs), np.array(ys), r


def predict_at_start(model):
    x = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        return float(model(x).item())


env = PointObstacleEnv()
single = collect(env, "left", 100)
multi = collect(env, "left", 50) + collect(env, "right", 50)

print("=== Q3 多模态崩溃：单模态 vs 多模态示范 ===")
results = {}
models = {}
for name, demos in [("single-mode", single), ("multi-mode", multi)]:
    states, actions = to_arrays(demos)
    per_succ, per_start = [], []
    for seed in SEEDS:
        m = train_bc(states, actions, seed)
        _, _, r = rollout(m, env)
        per_succ.append(1 if r > 0 else 0)
        per_start.append(predict_at_start(m))
        if seed == 0:
            models[name] = m
    ms = float(np.mean(per_succ))
    ss = float(np.std(per_succ))
    ma = float(np.mean(per_start))
    results[name] = (ma, ms, ss)
    print(f"{name:11s} | 起点动作 u(0,0)={ma:7.3f} | 成功率={ms:.2f} +/- {ss:.2f}")

with open("q3_multimodal_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "action_at_start", "success_rate", "success_std"])
    for name, (ma, ms, ss) in results.items():
        w.writerow([name, ma, ms, ss])

fig, ax = plt.subplots(figsize=(5, 6))
ax.add_patch(Rectangle((-0.2, 0.4), 0.4, 0.3, color="red", alpha=0.35))
for _ in range(3):
    xs, ys, _ = rollout(models["single-mode"], env)
    ax.plot(xs, ys, "g-", linewidth=2)
    xs, ys, _ = rollout(models["multi-mode"], env)
    ax.plot(xs, ys, "r--", linewidth=2)
ax.plot([], [], "g-", label="single-mode BC (绕行成功)")
ax.plot([], [], "r--", label="multi-mode BC (平均化撞墙)")
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(0, 1.05)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Multimodal ambiguity: BC averages two modes")
ax.legend()
plt.tight_layout()
plt.savefig("q3_multimodal_trajectories.png", dpi=150)

print("\nSaved q3_multimodal_results.csv and q3_multimodal_trajectories.png")