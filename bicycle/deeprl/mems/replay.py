from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
from mems.SumTree import *
import random

class Memory(object):

    def __init__(self, memory_size, state_dim, action_dim, batch_size):
        """
        Args
            memory_size: size of memory replay
            state_dim: state dimension
            action_dim: action dimension
            batch_size: number of samples which is sampled for training
        """
        self.memory_size = memory_size
        self.batch_size = batch_size

        if type(state_dim) is not tuple:
            state_dim = (state_dim, )

        # current state
        self.curr_state = np.empty(shape=(memory_size, ) + state_dim, dtype=np.float32)
        # next state
        self.next_state = np.empty(shape=(memory_size, ) + state_dim, dtype=np.float32)
        # reward
        self.rewards = np.empty(memory_size, dtype=np.float32)
        # terminal
        self.terminals = np.empty(memory_size, dtype=np.float32)
        # actions
        self.actions = np.empty((memory_size, action_dim) if action_dim > 1 else memory_size, dtype=np.float32)

        self.current = 0
        self.count = 0

    def add(self, sample, error=None):
        self.curr_state[self.current, ...] = sample[0]
        self.next_state[self.current, ...] = sample[1]
        self.rewards[self.current] = sample[2]
        self.terminals[self.current] = sample[3]
        self.actions[self.current] = sample[4]

        self.current += 1
        self.count = max(self.count, self.current)
        if self.current >= self.memory_size - 1:
            self.current = 0

    def sample(self):
        indexes = np.random.randint(0, self.count, self.batch_size)

        curr_state = self.curr_state[indexes, ...]
        next_state = self.next_state[indexes, ...]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        actions = self.actions[indexes]
        return [curr_state, next_state, rewards, terminals, actions], None

    def save(self, save_dir):
        path = os.path.join(save_dir, type(self).__name__)
        if not os.path.exists(path):
            os.makedirs(path)
        print("Saving memory...")
        for name in ("curr_state", "next_state", "rewards", "terminals", "actions"):
            np.save(os.path.join(path, name), arr=getattr(self, name))

    def update(self, idxs=None, errors=None):
        return

    def get_curr_size(self):
        return self.count

    def restore(self, save_dir):
        """
        Restore the memory.
        """
        path = os.path.join(save_dir, type(self).__name__)
        for name in ("curr_state", "next_state", "rewards", "terminals", "actions"):
            setattr(self, name, np.load(os.path.join(path, "%s.npy" % name)))

    def size(self):
        for name in ("curr_state", "next_state", "rewards", "terminals", "actions"):
            print("%s size is %s" % (name, getattr(self, name).shape))