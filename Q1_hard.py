import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

LENGTH = 0.3

env = gym.make("CartPole-v1")
env.unwrapped.length = LENGTH
env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length

model = PPO("MlpPolicy", env, seed=1, verbose=1)
model.learn(total_timesteps=150000)
model.save(f"ppo_cartpole_hard_{LENGTH}")

mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=30)
print(f"\nHard CartPole (length={LENGTH}) expert: mean reward = {mean_r:.1f} +/- {std_r:.1f}")