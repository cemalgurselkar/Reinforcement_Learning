import gymnasium as gym
import numpy as np


class Sarsa:
    def __init__(self,
                 env,
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.2,
                 eps_min=0.01,
                 eps_decay=0.997):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.q = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, s):
        if np.random.rand() < self.eps:
            return self.env.action_space.sample()
        return int(np.argmax(self.q[s]))

    def learn(self, s, a, r, ns, na, done):
        target = r if done else r + self.gamma * self.q[ns, na]
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def train(self, episodes, max_steps=500, log_every=100):
        returns = []
        for ep in range(1, episodes + 1):
            s, _ = self.env.reset()
            a = self.act(s)
            ret = 0.0
            for _ in range(max_steps):
                ns, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                na = self.act(ns)
                self.learn(s, a, r, ns, na, done)
                s, a = ns, na
                ret += r
                if done:
                    break
            returns.append(ret)

            self.eps = max(self.eps_min, self.eps * self.eps_decay)

            if ep % log_every == 0:
                avg = np.mean(returns[-log_every:])
                print(f"Episode {ep:4d} | avg return: {avg:6.3f} | ε={self.eps:.3f}")

        return returns

    def greedy_action(self, s):
        return int(np.argmax(self.q[s]))


def evaluate(agent, env, n_ep=20, max_steps=5000, render=False):
    for ep in range(n_ep):
        s, _ = env.reset()
        ret = 0.0
        for _ in range(max_steps):
            if render:
                env.render()
            s, r, term, trunc, _ = env.step(agent.greedy_action(s))
            ret += r
            if term or trunc:
                break
        
        print(f"Eval episode {ep+1: 2d} | return: {ret:.1f}")


if __name__ == "__main__":
    TRAIN_EPISODES = 500
    ALPHA = 0.1
    GAMMA = 0.99
    EPS = 0.2
    EPS_MIN = 0.01
    EPS_DECAY = 0.997
    RENDER_TRAIN = False        # eğitimde görmek istersen True yap
    RENDER_EVAL = True          # testte görelim

    env = gym.make("CliffWalking-v1", render_mode="human")
    agent = Sarsa(env, alpha=ALPHA, gamma=GAMMA,
                  epsilon=EPS, eps_min=EPS_MIN, eps_decay=EPS_DECAY)

    agent.train(TRAIN_EPISODES, log_every=100)

    input("Eğitim bitti. Test başlıyor – Enter'a bas...")
    evaluate(agent, env, n_ep=5, render=RENDER_EVAL)