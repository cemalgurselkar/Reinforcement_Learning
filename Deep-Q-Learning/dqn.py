import math, random
from collections import deque, namedtuple
from itertools import count

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

Transition = namedtuple("Transition", ("state","action","next_state","reward","done"))

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, n_actions)
        for m in (self.l1, self.l2, self.l3):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

class Agent:
    def __init__(self, obs_dim, n_actions,
                 gamma=0.99, lr=3e-4, tau=0.005,
                 batch_size=128, buffer_capacity=10000,
                 eps_start=0.9, eps_end=0.01, eps_decay=2500,
                 grad_clip=100.0, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma, self.tau = gamma, tau
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.n_actions = n_actions

        self.policy = QNet(obs_dim, n_actions).to(self.device)
        self.target = QNet(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optim = torch.optim.AdamW(self.policy.parameters(), lr=lr, amsgrad=True)
        self.buf = deque(maxlen=buffer_capacity)

        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.steps = 0

        self.loss_log = []
        self.eps_log = []

    def _eps(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.steps / self.eps_decay)

    def act(self, state_np: np.ndarray) -> int:
        eps = self._eps()
        self.eps_log.append(eps)
        self.steps += 1

        if random.random() > eps:
            with torch.no_grad():
                s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                return int(self.policy(s).argmax(dim=1).item())
        return random.randrange(self.n_actions)

    def remember(self, s, a, s_next, r, done) -> None:
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_t = torch.tensor([[a]], dtype=torch.long, device=self.device)
        ns_t = None if s_next is None else torch.as_tensor(s_next, dtype=torch.float32, device=self.device).unsqueeze(0)
        r_t = torch.tensor([r], dtype=torch.float32, device=self.device)
        d_t = torch.tensor([float(done)], dtype=torch.float32, device=self.device)
        self.buf.append(Transition(s_t, a_t, ns_t, r_t, d_t))

    def train_step(self):
        if len(self.buf) < self.batch_size:
            return

        batch = random.sample(self.buf, self.batch_size)
        s = torch.cat([t.state for t in batch])
        a = torch.cat([t.action for t in batch])
        r = torch.cat([t.reward for t in batch]).view(-1)
        d = torch.cat([t.done for t in batch]).view(-1)

        non_final = [t.next_state for t in batch if t.next_state is not None]
        mask = torch.tensor([t.next_state is not None for t in batch], device=self.device, dtype=torch.bool)
        if non_final:
            ns = torch.cat(non_final)

        q_sa = self.policy(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            next_vals = torch.zeros(self.batch_size, device=self.device)
            if non_final:
                next_vals[mask] = self.target(ns).max(1).values
            y = r + (1.0 - d) * self.gamma * next_vals

        loss = F.smooth_l1_loss(q_sa, y)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100.0)
        self.optim.step()
        self.loss_log.append(float(loss.item()))

    @torch.no_grad()
    def soft_update(self):
        for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * pp.data)

def train(env, agent: Agent, episodes: int):
    ep_lengths, ep_returns = [], []
    for _ in range(episodes):
        s, _ = env.reset()
        ep_len, ep_ret = 0, 0.0
        for _ in count():
            a = agent.act(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            agent.remember(s, a, (None if done else s_next), r, done)
            s = s_next
            ep_ret += r
            ep_len += 1

            agent.train_step()
            agent.soft_update()

            if done:
                ep_lengths.append(ep_len)
                ep_returns.append(ep_ret)
                break
    return ep_lengths, ep_returns

def evaluate(env, agent: Agent, episodes=5):
    total = 0.0
    for _ in range(episodes):
        s, _ = env.reset()
        ep_ret = 0.0
        while True:
            with torch.no_grad():
                st = torch.as_tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)
                a = int(agent.policy(st).argmax(dim=1).item())
            s, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            if terminated or truncated:
                total += ep_ret
                break
    return total / episodes

# Backend'i pyplot'tan ÖNCE seç
def _pick_backend():
    try:
        matplotlib.use("TkAgg", force=True)
        return True
    except Exception:
        matplotlib.use("Agg", force=True)
        return False

INTERACTIVE = _pick_backend()

def moving_avg(x, k):
    if len(x) < k: 
        return np.array(x, dtype=float)
    c = np.cumsum(np.insert(np.array(x, dtype=float), 0, 0.0))
    return (c[k:] - c[:-k]) / k

def _finalize(fig, path: str | None):
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if INTERACTIVE:
        plt.show()
    plt.close(fig)

def plot_all(ep_lengths, ep_returns, loss_log=None, eps_log=None, save_prefix="rl_plots", save_always=True):

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.set_title("Episode Length")
    ax.plot(ep_lengths, label="length")
    if len(ep_lengths) >= 20:
        ma = moving_avg(ep_lengths, 20)
        ax.plot(range(19, len(ep_lengths)), ma, label="MA(20)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps"); ax.legend()
    _finalize(fig, f"{save_prefix}_length.png" if (save_always or not INTERACTIVE) else None)

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.set_title("Episode Return")
    ax.plot(ep_returns, label="return")
    if len(ep_returns) >= 20:
        ma = moving_avg(ep_returns, 20)
        ax.plot(range(19, len(ep_returns)), ma, label="MA(20)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Return"); ax.legend()
    _finalize(fig, f"{save_prefix}_return.png" if (save_always or not INTERACTIVE) else None)

    if loss_log:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        ax.set_title("Training Loss")
        ax.plot(loss_log, label="loss")
        if len(loss_log) >= 200:
            ma = moving_avg(loss_log, 200)
            ax.plot(range(199, len(loss_log)), ma, label="MA(200)")
        ax.set_xlabel("Update step"); ax.set_ylabel("Loss"); ax.legend()
        _finalize(fig, f"{save_prefix}_loss.png" if (save_always or not INTERACTIVE) else None)

    if eps_log:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        ax.set_title("Epsilon (ε)")
        ax.plot(eps_log)
        ax.set_xlabel("Env step"); ax.set_ylabel("ε")
        _finalize(fig, f"{save_prefix}_epsilon.png" if (save_always or not INTERACTIVE) else None)

def main():

    train_env = gym.make("CartPole-v1", render_mode="human")
    obs_dim = int(np.array(train_env.reset()[0]).shape[0])
    n_actions = train_env.action_space.n

    agent = Agent(obs_dim, n_actions,
                  gamma=0.99, lr=3e-4, tau=0.005,
                  batch_size=128, buffer_capacity=10000,
                  eps_start=0.9, eps_end=0.01, eps_decay=2500)

    episodes = 600 if torch.cuda.is_available() else 50

    try:
        ep_lengths, ep_returns = train(train_env, agent, episodes)
    except KeyboardInterrupt:

        ep_lengths, ep_returns = ep_lengths if 'ep_lengths' in locals() else [], ep_returns if 'ep_returns' in locals() else []
    finally:
        train_env.close()
    
    plot_all(ep_lengths, ep_returns, agent.loss_log, agent.eps_log)

    eval_env = gym.make("CartPole-v1", render_mode=None)
    avg_ret = evaluate(eval_env, agent, episodes=5)
    eval_env.close()
    print(f"Eval avg return (5 eps): {avg_ret:.1f}")

if __name__ == "__main__":
    main()
