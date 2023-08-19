class Episode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

    def __len__(self):
        return len(self.observations)
