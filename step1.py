import gymnasium as gym

# 1. Create the environment
env = gym.make("CartPole-v1")

# 2. Look at the interface
print("=== 1. Basic info ===")
print("Observation space:", env.observation_space)       # 观测空间,有四个状态参量
print("Action space:", env.action_space)                 # 动作空间，0左力，1右力

# 3. Reset, get initial state
observation, info = env.reset(seed=42)
print("\n=== 2. Initial state ===")
print("Initial observation (4 numbers):", observation)

# 4. Random actions for 20 steps
print("\n=== 3. Random agent for 20 steps ===")
for t in range(20):
    action = env.action_space.sample()                  # random: 0 or 1
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"step {t:2d} | action={action} | reward={reward} | state={observation}")
    if terminated or truncated:
        print(f"-> ended at step {t} (terminated={terminated}, truncated={truncated})")
        print("a random policy can't keep the pole up for long")
        break

env.close()
print("\nDone.")