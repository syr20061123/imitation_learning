# -*- coding: utf-8 -*-
import csv
import numpy as np
np.float_ = np.float64
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_DEMOS = 100
SEEDS = list(range(8))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def collect(model, env, n=N_DEMOS, seed0=0):
    eps = []
    for i in range(n):
        obs, info = env.reset(seed=seed0 + i)
        ep = []
        done = False
        while not done:
            a = float(model.predict(obs, deterministic=True)[0].item())
            ep.append((obs.copy(), a))
            obs, r, term, trunc, info = env.step(np.array([a], np.float32))
            done = term or trunc
        eps.append(ep)
    return eps


def to_arrays(eps):
    s, a = [], []
    for ep in eps:
        for obs, act in ep:
            s.append(obs)
            a.append(act)
    return np.array(s, np.float32), np.array(a, np.float32).reshape(-1, 1)


def train_bc(states, actions, seed, epochs=100):
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


def evaluate_bc(model, env, n_ep=30, start=3000):
    rs = []
    for i in range(n_ep):
        obs, info = env.reset(seed=start + i)
        t = 0.0
        done = False
        while not done:
            x = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                a = float(model(x).item())
            a = float(np.clip(a, -2.0, 2.0))
            obs, r, term, trunc, info = env.step(np.array([a], np.float32))
            t += r
            done = term or trunc
        rs.append(t)
    return float(np.mean(rs)), float(np.std(rs))


env = gym.make("Pendulum-v1")
m1 = PPO.load("ppo_pendulum.zip", env=env)
m2 = PPO.load("ppo_pendulum_seed2.zip", env=env)

d1 = collect(m1, env, N_DEMOS)
d2 = collect(m2, env, N_DEMOS)

groups = [
    ("single-mode(exp1)", d1),
    ("single-mode(exp2)", d2),
    ("multi-mode(mix)", d1[:50] + d2[:50]),
]

print(f"=== Q3 Pendulum 多模态对照（N={N_DEMOS}）===")
results = {}
for name, demos in groups:
    states, actions = to_arrays(demos)
    per = []
    for seed in SEEDS:
        model = train_bc(states, actions, seed)
        m, _ = evaluate_bc(model, env)
        per.append(m)
    mean_r, std_r = float(np.mean(per)), float(np.std(per))
    results[name] = (mean_r, std_r)
    print(f"{name:20s} | BC reward = {mean_r:8.1f} +/- {std_r:6.1f}")

with open("q3_pendulum_multimodal_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "mean_reward", "std"])
    for name, (mean_r, std_r) in results.items():
        w.writerow([name, mean_r, std_r])

labels = list(results.keys())
x = np.arange(len(labels))
plt.figure(figsize=(7, 4.5))
plt.bar(x, [results[n][0] for n in labels],
        yerr=[results[n][1] for n in labels], capsize=4)
plt.xticks(x, labels)
plt.ylabel("BC mean reward")
plt.title("Q3 Pendulum: single-mode vs multi-mode")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("q3_pendulum_multimodal_bar.png", dpi=150)

print("\nSaved q3_pendulum_multimodal_results.csv and q3_pendulum_multimodal_bar.png")