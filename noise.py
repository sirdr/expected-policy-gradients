import numpy as np



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed=7, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class NormalActionNoise:
    def __init__(self, mu, sigma, size):
        self.mu = mu
        self.sigma = sigma
        self.size = size

    def sample(self):
        return np.random.normal(self.mu, self.sigma, size=self.size)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    def reset(self):
        pass
