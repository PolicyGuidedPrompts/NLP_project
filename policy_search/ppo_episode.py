class PPOEpisode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.old_logprobs = []

    def add(self, observation, action, reward, old_logprob):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.old_logprobs.append(old_logprob)

    def __len__(self):
        return len(self.observations)
