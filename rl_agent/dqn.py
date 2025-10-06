# rl_agent/dqn.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque

class Net(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, done = zip(*batch)
        return np.stack(s), np.array(a), np.array(r), np.stack(ns), np.array(done)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim=2, device=None, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.qnet = Net(state_dim, action_dim).to(self.device)
        self.target = Net(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.gamma = gamma
        self.replay = ReplayBuffer(20000)
        self.update_count = 0

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.qnet(s)
        return int(q.argmax().item())

    def remember(self, s, a, r, ns, done):
        self.replay.push(s, a, r, ns, done)

    def soft_update(self, tau=0.01):
        for p, q in zip(self.target.parameters(), self.qnet.parameters()):
            p.data.copy_(p.data * (1 - tau) + q.data * tau)

    def train_step(self, batch_size=64):
        if len(self.replay) < batch_size:
            return 0.0
        s, a, r, ns, done = self.replay.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.qnet(s)
        q = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target(ns).max(1)[0]
            target = r + (1 - done) * self.gamma * q_next

        loss = nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % 10 == 0:
            self.soft_update(tau=0.05)
        return loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.qnet.state_dict(), path)

    @staticmethod
    def load(path, state_dim, action_dim=2, device=None):
        agent = DQNAgent(state_dim, action_dim, device)
        agent.qnet.load_state_dict(torch.load(path, map_location=agent.device))
        agent.target.load_state_dict(agent.qnet.state_dict())
        return agent
