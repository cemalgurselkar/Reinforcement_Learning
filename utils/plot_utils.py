import matplotlib.pyplot as plt
from typing import List
import numpy as np
import os


def plot_rewards(rewards: List[float], outdir: str):
    plt.figure()
    plt.plot(rewards, linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rewards.png"))
    plt.close()

def plot_moving_avg(rewards: List[float], window: int, outdir: str):
    if len(rewards) < window:
        return
    csum = np.cumsum(np.insert(rewards, 0, 0))
    ma = (csum[window:] - csum[:-window]) / float(window)
    plt.figure()
    plt.plot(np.arange(window, window + len(ma)), ma, linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel(f"Avg Reward ({window})")
    plt.title(f"Moving Average Reward (window={window})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"moving_avg_{window}.png"))
    plt.close()

def plot_epsilon(epsilons: List[float], outdir: str):
    plt.figure()
    plt.plot(epsilons, linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "epsilon.png"))
    plt.close()

def plot_episode_lengths(lengths: List[int], outdir: str):
    plt.figure()
    plt.plot(lengths, linewidth=1.0)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Lengths")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "episode_lengths.png"))
    plt.close()

def plot_policy_grid_if_applicable(q_table: np.ndarray, env_name: str, state_count: int, outdir: str):
    # FrozenLake 4x4 için basit 2D policy görselleştirme
    side = int(np.sqrt(state_count))
    if env_name.startswith("FrozenLake") and side * side == state_count:
        policy = np.argmax(q_table, axis=1).reshape(side, side)
        plt.figure()
        im = plt.imshow(policy, interpolation="nearest")
        plt.title("Learned Policy (0:← 1:↓ 2:→ 3:↑)")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "policy_grid.png"))
        plt.close()

def finalize_plots(rewards, epsilons, ep_lengths, q_table, env_name, outdir):
    os.makedirs(outdir, exist_ok=True)
    plot_rewards(rewards, outdir)
    plot_moving_avg(rewards, window=100, outdir=outdir)
    plot_epsilon(epsilons, outdir)
    plot_episode_lengths(ep_lengths, outdir)
    plot_policy_grid_if_applicable(q_table, env_name, q_table.shape[0], outdir)

    # basit metrik özeti
    avg_last_100 = float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards))
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Episodes: {len(rewards)}\n")
        f.write(f"Avg reward (last 100): {avg_last_100:.4f}\n")
        f.write(f"Final epsilon: {epsilons[-1] if epsilons else 'n/a'}\n")