import numpy as np
import matplotlib

# Backend'i pyplot'tan ÖNCE seç
def _pick_backend():
    try:
        import tkinter  # GUI var mı?
        matplotlib.use("TkAgg", force=True)
        return True
    except Exception:
        matplotlib.use("Agg", force=True)
        return False

INTERACTIVE = _pick_backend()

import matplotlib.pyplot as plt  # backend seçildikten sonra import!

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
    plt.close(fig)  # bellek sızıntısını önle

def plot_all(ep_lengths, ep_returns, loss_log=None, eps_log=None, save_prefix="rl_plots", save_always=True):
    # 1) Episode length
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.set_title("Episode Length")
    ax.plot(ep_lengths, label="length")
    if len(ep_lengths) >= 20:
        ma = moving_avg(ep_lengths, 20)
        ax.plot(range(19, len(ep_lengths)), ma, label="MA(20)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps"); ax.legend()
    _finalize(fig, f"{save_prefix}_length.png" if (save_always or not INTERACTIVE) else None)

    # 2) Episode return
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.set_title("Episode Return")
    ax.plot(ep_returns, label="return")
    if len(ep_returns) >= 20:
        ma = moving_avg(ep_returns, 20)
        ax.plot(range(19, len(ep_returns)), ma, label="MA(20)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Return"); ax.legend()
    _finalize(fig, f"{save_prefix}_return.png" if (save_always or not INTERACTIVE) else None)

    # 3) Loss
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

    # 4) Epsilon
    if eps_log:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        ax.set_title("Epsilon (ε)")
        ax.plot(eps_log)
        ax.set_xlabel("Env step"); ax.set_ylabel("ε")
        _finalize(fig, f"{save_prefix}_epsilon.png" if (save_always or not INTERACTIVE) else None)
