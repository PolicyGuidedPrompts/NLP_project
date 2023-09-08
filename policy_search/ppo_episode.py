class PPOEpisode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.old_logprobs = []
        self.total_reward = 0

    def add(self, observation, action, reward, old_logprob):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.old_logprobs.append(old_logprob)
        self.total_reward += reward

    def __len__(self):
        return len(self.observations)
