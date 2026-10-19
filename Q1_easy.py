import pickle
import csv
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("demos_expert.pkl", "rb") as f:
    all_episodes = pickle.load(f)
print(f"Total demonstrations available: {len(all_episodes)}")

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

def episodes_to_arrays(episodes):
    states, actions = [], []
    for ep in episodes:
        for obs, act in ep:
            states.append(obs); actions.append(act)
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)

def sample_episodes(all_episodes, n, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_episodes), size=n, replace=False)
    return [all_episodes[i] for i in idx]

def train_bc(states, actions, seed, epochs=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    X = torch.from_numpy(states); Y = torch.from_numpy(actions)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            logits = model(X[idx])
            loss = loss_fn(logits, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def evaluate(model, env, n_episodes=30, seed_start=1000):
    rewards = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed_start + i)
        total = 0.0; done = False
        while not done:
            x = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
            action = int(torch.argmax(logits, dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward; done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


env = gym.make("CartPole-v1")
N_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]
TRAIN_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

results = {}
for N in N_values:
    per_seed = []
    for seed in TRAIN_SEEDS:
        episodes = sample_episodes(all_episodes, N, seed)
        states, actions = episodes_to_arrays(episodes)
        model = train_bc(states, actions, seed)
        mean_r, _ = evaluate(model, env)
        per_seed.append(mean_r)
    m, s = float(np.mean(per_seed)), float(np.std(per_seed))
    results[N] = (m, s)
    print(f"N = {N:3d} demos -> BC mean reward = {m:6.1f} +/- {s:5.1f}")

with open("q1_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N_demos", "mean_reward", "std"])
    for n in N_values:
        w.writerow([n, results[n][0], results[n][1]])

plt.figure(figsize=(7, 4.5))
plt.errorbar(N_values, [results[n][0] for n in N_values],
             yerr=[results[n][1] for n in N_values], marker="o", capsize=4)
plt.xscale("log")
plt.xlabel("Number of demonstrations (log scale)")
plt.ylabel("BC mean reward")
plt.title("Q1: BC performance vs number of demonstrations")
plt.ylim(0, 550)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("q1_quantity.png", dpi=150)
print("\nSaved q1_quantity.png and q1_results.csv")