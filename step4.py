import pickle
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

with open("demos_expert.pkl", "rb") as f:
    episodes = pickle.load(f)

states, actions = [], []
for ep in episodes:
    for obs, act in ep:
        states.append(obs)
        actions.append(act)
states = np.array(states, dtype=np.float32)
actions = np.array(actions, dtype=np.int64)
print(f"Training samples: {len(states)}")

class MLP(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

X = torch.from_numpy(states)
Y = torch.from_numpy(actions)

EPOCHS = 50
BATCH = 256
for epoch in range(EPOCHS):
    perm = torch.randperm(len(X))
    total_loss = 0.0
    for i in range(0, len(X), BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = X[idx], Y[idx]
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch + 1:3d} | loss = {total_loss:.4f}")

env = gym.make("CartPole-v1")

def evaluate_bc(model, n_episodes=30, seed=100):
    rewards = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        total = 0
        done = False
        while not done:
            x = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
            action = int(torch.argmax(logits, dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)

mean_r, std_r = evaluate_bc(model, n_episodes=30)
print(f"\nBC policy performance: mean reward = {mean_r:.1f} +/- {std_r:.1f}")

torch.save(model.state_dict(), "bc_model.pt")
print("Saved bc_model.pt")