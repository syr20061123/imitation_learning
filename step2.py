import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Create the environment (8 parallel copies = faster training)
env = make_vec_env("CartPole-v1", n_envs=8, seed=0)

# 2. Create the PPO agent (MLP policy)
model = PPO("MlpPolicy", env, seed=1, verbose=1)

# 3. Train it (about 1-2 minutes on CPU)
model.learn(total_timesteps=300000)

# 4. Save the expert for later use
model.save("ppo_cartpole")

# 5. Evaluate: 30 episodes, report average score
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30)
print(f"\nExpert performance: mean reward = {mean_reward:.1f} +/- {std_reward:.1f}")
print("(Perfect score = 500, the max steps per episode)")