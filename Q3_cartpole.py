# -*- coding: utf-8 -*-
import csv
import numpy as np
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
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def set_state(env, st):
    env.unwrapped.state = np.array(st, np.float32)
    return env.unwrapped.state.copy()


def similar_start(rng):
    return np.array([rng.uniform(-0.05, 0.05), 0.0,
                     rng.uniform(-0.02, 0.02), 0.0], np.float32)


def diverse_start(rng):
    return np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.2, 0.2),
                     rng.uniform(-0.12, 0.12), rng.uniform(-0.2, 0.2)], np.float32)


def collect(expert, env, n=N_DEMOS, mode="similar", seed0=0):
    rng = np.random.RandomState(seed0)
    eps = []
    for i in range(n):
        obs, info = env.reset(seed=seed0 + i)
        st = similar_start(rng) if mode == "similar" else diverse_start(rng)
        obs = set_state(env, st)
        ep = []
        done = False
        while not done:
            a = int(expert.predict(obs, deterministic=True)[0].item())
            ep.append((obs.copy(), a))
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        eps.append(ep)
    return eps


def compute_state_diversity(eps):
    starts = np.array([ep[0][0] for ep in eps], dtype=np.float32)
    return float(np.sum(np.var(starts, axis=0)))


def to_arrays(eps):
    s, a = [], []
    for ep in eps:
        for obs, act in ep:
            s.append(obs)
            a.append(act)
    return np.array(s, np.float32), np.array(a, np.int64)


def train_bc(states, actions, seed, epochs=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = MLP()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
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


def evaluate_bc(model, env, n_ep=30, start=3000, mode="wide"):
    rng = np.random.RandomState(start)
    rs = []
    for i in range(n_ep):
        obs, info = env.reset(seed=start + i)
        st = similar_start(rng) if mode == "narrow" else diverse_start(rng)
        obs = set_state(env, st)
        t = 0.0
        done = False
        while not done:
            x = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
            a = int(torch.argmax(logits, 1).item())
            obs, r, term, trunc, info = env.step(a)
            t += r
            done = term or trunc
        rs.append(t)
    return float(np.mean(rs)), float(np.std(rs))


env = gym.make("CartPole-v1")
expert = PPO.load("ppo_cartpole.zip", env=env)

print(f"=== Q3 (CartPole): diversity vs generalization (N={N_DEMOS}) ===")
results = {}
for name in ["similar", "diverse"]:
    eps = collect(expert, env, mode=name)
    div = compute_state_diversity(eps)
    states, actions = to_arrays(eps)
    per_nar, per_wide = [], []
    for seed in SEEDS:
        model = train_bc(states, actions, seed)
        nar, _ = evaluate_bc(model, env, mode="narrow")
        wid, _ = evaluate_bc(model, env, mode="wide")
        per_nar.append(nar)
        per_wide.append(wid)
    mn, sn = float(np.mean(per_nar)), float(np.std(per_nar))
    mw, sw = float(np.mean(per_wide)), float(np.std(per_wide))
    results[name] = (div, mn, sn, mw, sw)
    print(f"{name:8s} | 状态多样性={div:7.4f} | "
          f"BC 窄起点={mn:6.1f} +/- {sn:4.1f} | BC 宽起点={mw:6.1f} +/- {sw:4.1f}")

with open("q3_cartpole_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "state_diversity", "bc_narrow_mean", "bc_narrow_std",
                "bc_wide_mean", "bc_wide_std"])
    for name, (div, mn, sn, mw, sw) in results.items():
        w.writerow([name, div, mn, sn, mw, sw])

labels = list(results.keys())
x = np.arange(len(labels))
wbar = 0.35
plt.figure(figsize=(7, 4.5))
plt.bar(x - wbar/2, [results[n][1] for n in labels], wbar,
        yerr=[results[n][2] for n in labels], capsize=4, label="BC on narrow start")
plt.bar(x + wbar/2, [results[n][3] for n in labels], wbar,
        yerr=[results[n][4] for n in labels], capsize=4, label="BC on wide start")
plt.xticks(x, labels)
plt.ylabel("BC mean reward")
plt.ylim(0, 550)
plt.title("Q3 CartPole: diversity has little effect")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("q3_cartpole_bar.png", dpi=150)

print("\nSaved q3_cartpole_results.csv and q3_cartpole_bar.png")