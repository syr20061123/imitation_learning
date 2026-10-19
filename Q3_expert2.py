# -*- coding: utf-8 -*-
import os
import numpy as np
np.float_ = np.float64
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

SEED2_PATH = "ppo_pendulum_seed2.zip"

env = gym.make("Pendulum-v1")

if not os.path.exists(SEED2_PATH):
    print("training expert2 (seed=2, 1M steps)...")
    m2 = PPO("MlpPolicy", env, seed=2, verbose=0)
    m2.learn(total_timesteps=1_000_000)
    m2.save(SEED2_PATH)
else:
    m2 = PPO.load(SEED2_PATH, env=env)

m1 = PPO.load("ppo_pendulum.zip", env=env)

r1, s1 = evaluate_policy(m1, env, n_eval_episodes=30)
r2, s2 = evaluate_policy(m2, env, n_eval_episodes=30)
print(f"expert1 reward = {r1:.1f} +/- {s1:.1f}")
print(f"expert2 reward = {r2:.1f} +/- {s2:.1f}")


def observe_direction(model, env):
    # 从倒挂点 (pi, 0) 出发
    env.reset(seed=0)
    env.unwrapped.state = np.array([np.pi, 0.0])
    obs = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    a0 = float(model.predict(obs, deterministic=True)[0].item())
    tds = []
    for _ in range(30):
        a = float(model.predict(obs, deterministic=True)[0].item())
        a = float(np.clip(a, -2.0, 2.0))
        obs, r, term, trunc, info = env.step(np.array([a], np.float32))
        tds.append(env.unwrapped.state[1])
    return a0, float(np.mean(tds))


a1, d1 = observe_direction(m1, env)
a2, d2 = observe_direction(m2, env)
print(f"expert1: action@(pi,0)={a1:.2f}, 前30步平均角速度={d1:+.2f}")
print(f"expert2: action@(pi,0)={a2:.2f}, 前30步平均角速度={d2:+.2f}")
print("若两者平均角速度异号，说明两个专家摆向相反；同号则摆向相同。")