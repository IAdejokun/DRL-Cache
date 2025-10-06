# rl_agent/train.py
import argparse
import time
import numpy as np
from rl_agent.env import CacheEnv
from rl_agent.dqn import DQNAgent

def train(args):
    env = CacheEnv(
        num_keys=args.num_keys,
        max_bytes=args.max_bytes,
        hit_latency_ms=args.hit_ms,
        miss_latency_range=(args.miss_min, args.miss_max),
        size_kb_range=(args.min_kb, args.max_kb),
        ttl_s=args.ttl_s,
        workload=args.workload,
        seed=args.seed,
    )
    state_dim = 5
    agent = DQNAgent(state_dim, action_dim=2, lr=args.lr)
    total_steps = 0

    epsilon_start = args.epsilon_start
    epsilon_final = args.epsilon_final
    epsilon_decay = args.epsilon_decay

    for ep in range(args.episodes):
        # reset env (not returning obs is ok)
        obs = env.reset()
        ep_reward = 0.0
        for t in range(args.steps_per_episode):
            epsilon = max(epsilon_final, epsilon_start * (epsilon_decay ** (ep)))
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, done, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done)
            loss = agent.train_step(batch_size=args.batch_size)
            obs = next_obs
            ep_reward += reward
            total_steps += 1
        if ep % args.log_every == 0:
            print(f"Episode {ep}/{args.episodes} reward={ep_reward:.3f} epsilon={epsilon:.3f} loss={loss:.6f}")
        if ep % args.save_every == 0 and ep > 0:
            agent.save(args.out)
            print(f"Saved model to {args.out} at episode {ep}")
    # final save
    agent.save(args.out)
    print("Training complete. Saved model to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps-per-episode", dest="steps_per_episode", type=int, default=200)
    p.add_argument("--num-keys", type=int, default=50)
    p.add_argument("--max-bytes", type=int, default=50_000_000)
    p.add_argument("--hit-ms", type=int, default=20)
    p.add_argument("--miss-min", type=int, default=100)
    p.add_argument("--miss-max", type=int, default=200)
    p.add_argument("--min-kb", type=int, default=5)
    p.add_argument("--max-kb", type=int, default=500)
    p.add_argument("--ttl-s", dest="ttl_s", type=int, default=300)
    p.add_argument("--workload", type=str, default="zipf")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--save-every", dest="save_every", type=int, default=50)
    p.add_argument("--log-every", dest="log_every", type=int, default=10)
    p.add_argument("--out", type=str, default="rl_agent/model.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-final", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    args = p.parse_args()
    train(args)
