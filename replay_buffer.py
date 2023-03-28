from collections import deque
import random


class ReplayBuffer():
    def __init__(self, size: int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)

    def push(self, transition) -> deque:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type

        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size: int) -> list:
        """Get a random sample from the replay buffer

        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)
