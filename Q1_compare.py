import pickle
import csv
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LENGTH = 0.3
HARD_MODEL = "ppo_cartpole_hard_0.3"


def make_hard_env():
    env = gym.make("CartPole-v1")
    env.unwrapped.length = LENGTH
    env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length
    return env


expert = PPO.load(HARD_MODEL)
env = make_hard_env()

episodes = []
for i in range(200):
    obs, info = env.reset(seed=i)
    ep = []
    done = False
    while not done:
        action = int(expert.predict(obs, deterministic=True)[0].item())
        ep.append((obs.copy(), action))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    episodes.append(ep)

lens = [len(e) for e in episodes]
print(f"Hard demos collected: {len(episodes)} episodes, avg length = {np.mean(lens):.1f}")
with open("demos_hard_expert.pkl", "wb") as f:
    pickle.dump(episodes, f)

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


eval_env = make_hard_env()
N_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]
TRAIN_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

results = {}
for N in N_values:
    per_seed = []
    for seed in TRAIN_SEEDS:
        eps = sample_episodes(episodes, N, seed)
        states, actions = episodes_to_arrays(eps)
        model = train_bc(states, actions, seed)
        mean_r, _ = evaluate(model, eval_env)
        per_seed.append(mean_r)
    m, s = float(np.mean(per_seed)), float(np.std(per_seed))
    results[N] = (m, s)
    print(f"hard N = {N:3d} -> BC mean reward = {m:6.1f} +/- {s:5.1f}")

with open("q1_hard_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N_demos", "mean_reward", "std"])
    for n in N_values:
        w.writerow([n, results[n][0], results[n][1]])

easy = {}
with open("q1_results.csv") as f:
    for row in csv.DictReader(f):
        easy[int(row["N_demos"])] = (float(row["mean_reward"]), float(row["std"]))

plt.figure(figsize=(7, 4.5))
plt.errorbar(list(easy.keys()), [easy[n][0] for n in easy],
             yerr=[easy[n][1] for n in easy], marker="o", capsize=4, label="easy (length=0.5)")
plt.errorbar(list(results.keys()), [results[n][0] for n in results],
             yerr=[results[n][1] for n in results], marker="s", capsize=4, label="hard (length=0.3)")
plt.xscale("log")
plt.xlabel("Number of demonstrations (log scale)")
plt.ylabel("BC mean reward")
plt.title("Q1: difficulty shift (easy vs hard CartPole)")
plt.ylim(0, 550)
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("q1_difficulty_shift.png", dpi=150)
print("\nSaved q1_difficulty_shift.png")