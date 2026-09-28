"""Replay buffer (RL, sliding window) and reservoir buffer (SL, uniform over all history)."""
import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", "state action reward next_state done mask next_mask")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class ReservoirBuffer:
    """Reservoir sampling: keeps a roughly uniform sample over the *entire* insertion
    history (not just recent items), matching the average-strategy update in fictitious
    play -- this is what makes the SL network approximate a time-average of past best
    responses rather than a recency-biased snapshot."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0

    def push(self, item):
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            idx = random.randrange(self.n_seen)
            if idx < self.capacity:
                self.buffer[idx] = item

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
