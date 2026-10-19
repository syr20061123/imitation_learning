import pickle
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "ppo_cartpole.zip"

def noisy_action(model, obs, sigma):
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model.policy.get_distribution(obs_t).distribution.logits
    noisy_logits = logits + sigma * torch.randn_like(logits)
    probs = torch.softmax(noisy_logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def collect_episode(env, model, sigma, seed):
    obs, info = env.reset(seed=seed)
    episode = []
    total_reward = 0
    done = False
    while not done:
        if sigma == 0:
            action = int(model.predict(obs, deterministic=True)[0].item())
        else:
            action = noisy_action(model, obs, sigma)
        episode.append((obs.copy(), action))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return episode, total_reward

def collect_dataset(env, model, n_episodes, sigma):
    episodes, rewards = [], []
    for i in range(n_episodes):
        ep, r = collect_episode(env, model, sigma, seed=i)
        episodes.append(ep)
        rewards.append(r)
    return episodes, rewards

def save_and_report(name, episodes, rewards):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(episodes, f)
    print(f"{name}: {len(episodes)} episodes | avg reward = {np.mean(rewards):6.1f} | "
          f"min = {np.min(rewards):4.0f} | max = {np.max(rewards):4.0f}")


if __name__ == "__main__":
    model = PPO.load(MODEL_PATH)
    env = gym.make("CartPole-v1")

    eps, rws = collect_dataset(env, model, 200, sigma=0.0)
    save_and_report("demos_expert", eps, rws)

    for sigma in [1.5, 2.0, 2.5, 3.0, 4.0]:
        eps, rws = collect_dataset(env, model, 200, sigma=sigma)
        save_and_report(f"demos_noisy_{sigma}", eps, rws)

    print("\nDone. Datasets saved as .pkl files.")