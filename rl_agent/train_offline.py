# rl_agent/train_offline.py
import argparse
import os
import csv
import time
import numpy as np
from rl_agent.trace_env import TraceEnv
from rl_agent.dqn import DQNAgent

def train(args):
    env = TraceEnv(csv_path=args.trace, max_bytes=args.max_bytes, ttl_s=args.ttl_s, hit_latency_ms=args.hit_ms, seed=args.seed)
    state_dim = 5
    agent = DQNAgent(state_dim, action_dim=2, lr=args.lr)
    total_steps = 0
    log_path = args.log or "logs/train_log.csv"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # CSV header
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "steps", "avg_reward", "epsilon", "loss"])

    for epoch in range(args.epochs):
        obs_arr, _ = env.reset() if isinstance(env.reset(), tuple) else env.reset()
        # env.reset returns (obs, meta) or obs; handle both
        if isinstance(obs_arr, tuple):
            obs = obs_arr[0]
        else:
            obs = obs_arr
        ep_reward = 0.0
        losses = []
        steps = 0

        # iterate through whole trace once per epoch
        done = False
        while not done:
            epsilon = max(args.epsilon_final, args.epsilon_start * (args.epsilon_decay ** epoch))
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, done, info = env.step(action)
            agent.remember(obs, action, reward, next_obs if next_obs is not None else np.zeros(state_dim), done)
            loss = agent.train_step(batch_size=args.batch_size)
            if loss:
                losses.append(loss)
            ep_reward += reward
            steps += 1
            total_steps += 1
            if next_obs is None:
                break
            obs = next_obs

        avg_loss = np.mean(losses) if losses else 0.0
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, steps, ep_reward / max(1, steps), float(epsilon), avg_loss])

        if epoch % args.save_every == 0 and epoch > 0:
            agent.save(args.out)
            print(f"Saved model checkpoint at epoch {epoch} to {args.out}")

        print(f"[Epoch {epoch}] reward={ep_reward:.3f} steps={steps} epsilon={epsilon:.3f} avg_loss={avg_loss:.6f}")

    # final save
    agent.save(args.out)
    print("Training finished. Model saved to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="Path to simulator CSV trace")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--save-every", dest="save_every", type=int, default=10)
    p.add_argument("--out", type=str, default="rl_agent/model.pt")
    p.add_argument("--max-bytes", type=int, default=50_000_000)
    p.add_argument("--ttl-s", type=int, default=300)
    p.add_argument("--hit-ms", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-final", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", type=str, default="logs/train_log.csv")
    args = p.parse_args()
    train(args)
