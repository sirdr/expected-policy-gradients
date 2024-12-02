import numpy as np

# Code based on: 
# https://github.com/sfujim/TD3/blob/master/utils.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        # Ensure all elements in data are consistent in type and shape
        X, Y, U, R, D = data
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        U = np.array(U, dtype=np.float32)
        R = float(R)  # Ensure scalar
        D = bool(D)   # Ensure boolean

        # Append to buffer or replace old data
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = (X, Y, U, R, D)
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append((X, Y, U, R, D))

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind: 
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)