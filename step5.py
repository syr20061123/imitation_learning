import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env

rng = np.random.default_rng(0)

expert = PPO.load("ppo_cartpole.zip")

venv = make_vec_env(
    "CartPole-v1",
    n_envs=1,
    rng=rng,
    post_wrappers=[lambda env, idx: RolloutInfoWrapper(env)],
)

rollouts = rollout.rollout(
    expert,
    venv,
    rollout.make_sample_until(min_timesteps=None, min_episodes=200),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)
print(f"Collected {len(transitions)} (state, action) pairs")

bc_trainer = bc.BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=50)

eval_env = gym.make("CartPole-v1")

def evaluate(policy, env, n_episodes=30, seed=300):
    rewards = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        total = 0.0
        done = False
        while not done:
            act, _ = policy.predict(obs, deterministic=True)
            act = int(np.asarray(act).reshape(-1)[0])
            obs, reward, terminated, truncated, info = env.step(act)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))

mean_r, std_r = evaluate(bc_trainer.policy, eval_env, n_episodes=30)
print(f"\nimitation BC: mean reward = {mean_r:.1f} +/- {std_r:.1f}")