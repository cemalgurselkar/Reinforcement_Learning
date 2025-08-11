import os
import sys
import signal
import argparse
from dataclasses import dataclass
from typing import List
from utils.plot_utils import *

import numpy as np
import keyboard

try:
    import gymnasium as gym
    NEW_GYM_API = True
except ImportError:
    import gym
    NEW_GYM_API = False

# -----Helpers Functions-------

def ensure_discrete_env(env):
    assert hasattr(env.observation_space, "n")
    assert hasattr(env.action_space, "n")
    return env.observation_space.n, env.action_space.n

def env_reset(env, seed=None):
    if NEW_GYM_API:
        obs, info = env.reset(seed=seed)
        return int(obs)
    else:
        obs = env.reset(seed=seed) if "seed" in env.reset.__code__.co_varnames else env.reset()
        return int(obs if not isinstance(obs, tuple) else obs[0])
    
def env_step(env, action):
    if NEW_GYM_API:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        return int(obs), float(reward), bool(done), info
    else:
        obs, reward, done, info = env.step(action)
        return int(obs), float(reward), bool(done), info


@dataclass
class QLearningConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.999
    epsilon_min:float = 0.01

class QLearningAgent:
    def __init__(self, n_states, n_action, config: QLearningConfig):
        self.n_states = n_states
        self.n_action = n_action
        self.q = np.zeros((n_states, n_action), dtype=np.float32) # initialize
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min

    def act(self, s) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_action)
        return int(np.argmax(self.q[s]))

    def learn(self, s, a, r, s_next, done):
        best_next = int(np.argmax(self.q[s_next]))
        td_target = r + (0.0 if done else self.gamma * self.q[s_next, best_next])
        td_error = td_target - self.q[s, a]
        self.q[s,a] += self.alpha * td_error

    def decay_eps(self, done):
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def greedy_action(self, s) -> int:
        return int(np.argmax(self.q[s]))

def train(env_name: str,
          episodes: int,
          max_steps: int,
          config: QLearningConfig,
          seed: int,
          render: bool,
          log_interval: int,
          outdir: str):
    
    env = gym.make(env_name, render_mode="human")
    n_states, n_actions = ensure_discrete_env(env)

    try:
        env.reset(seed=seed)
        np.random.seed(seed)
    except TypeError:
        np.random.seed(seed)
    
    agent = QLearningAgent(n_states, n_actions, config)

    reward: List[float] = []
    epsilons: List[float] = []
    ep_length: List[float] = []

    def handle_sigint(sig, frame):
        print("\n[!] Interrupted. Saving plots...")
        finalize_plots(reward, epsilons, ep_length, agent.q, env_name, outdir)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_sigint)
    os.makedirs(outdir, exist_ok=True)

    for ep in range(episodes):
        s = env_reset(env, seed=None)
        total_r = 0.0
        steps = 0

        for t in range(max_steps):
            
            if render:
                env.render()
            
            a = agent.act(s)
            s_next, r, done, _ = env_step(env, a)
            agent.learn(s, a, r, s_next, done)

            s = s_next
            total_r += r
            steps += 1

            if done:
                agent.decay_eps(done=True)
                break
        
        reward.append(total_r)
        epsilons.append(agent.epsilon)
        ep_length.append(steps)

        if (ep + 1) % log_interval == 0:
            avg100 = np.mean(reward[-min(100, len(reward)):])
            print(f"[{ep+1:5d}/{episodes}] avgR@100={avg100:.3f} eps={agent.epsilon:.3f} steps={steps}")
    
    env.close()
    finalize_plots(reward, epsilons, ep_length, agent.q, env_name, outdir)

def parse_args():
    p = argparse.ArgumentParser(description="Q-Learning Trainer")
    p.add_argument("--env", type=str, default="FrozenLake-v1",
                   help="Environment name (e.g., FrozenLake-v1, Taxi-v3, CliffWalking-v0)")
    p.add_argument("--episodes", type=int, default=3000, help="Number of training episodes")
    p.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    p.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    p.add_argument("--epsilon-decay", type=float, default=0.999, help="Epsilon decay per episode")
    p.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum epsilon")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--render", action="store_true", help="Render environment")
    p.add_argument("--log-interval", type=int, default=100, help="Logging interval (episodes)")
    p.add_argument("--outdir", type=str, default="plots", help="Directory to save plots & summary")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = QLearningConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )
    train(
        env_name=args.env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        config=cfg,
        seed=args.seed,
        render=args.render,
        log_interval=args.log_interval,
        outdir=args.outdir,
    )

if __name__ == "__main__":
    main()