import logging

import gym
from gym.spaces import Discrete, Box

# TODO - training_dataset + llm should be defined from configuration file
# TODO - Deberta tokenizer and model

# TODO - maybe train encoder as well
# TODO - speak with Nachum about Masters partition slurm

# TODO - add reward metric to config
# TODO - punish on long episodes

# TODO
# Roi suggestion
# choosing an action will actually choose a random vector
# will then choose the closest vector to that random vector

# Questions:
# Ask about budget

logger = logging.getLogger('root')


class Environment(gym.Env):
    def __init__(self, config, dataset, llm, encoder, seed, terminate_action=0):
        super(Environment, self).__init__()
        self.seed = seed
        self.config = config

        self.dataset = dataset
        self.encoder = encoder
        self.llm = llm

        self.terminate_action = terminate_action
        self.action_dim = len(self.dataset.data) + 1  # +1 for terminate action
        self.observation_dim = self.encoder.output_dimension

        self.reset()
        logger.info(f"Environment initialized with: "
                    f"({self.seed=}, "
                    f"{self.action_dim=}, "
                    f"{self.observation_dim=}, "
                    f"{self.llm.model_name=}, "
                    f"{self.encoder.model_name=})")

    def _update_prompts_based_on_actions(self, action):
        sampled_question, sampled_answer = self.dataset.data.iloc[action]
        self.question = f"Question: {sampled_question}\nAnswer: {sampled_answer}\n{self.question}"

    def step(self, actions):
        if actions == self.terminate_action:
            done = True
        else:
            self._update_prompts_based_on_actions(actions)
            done = self.llm.is_prompt_too_long(self.question)

        if done:
            reward, generated_answer = self.evaluate_prompt()
        else:
            reward = 0
            generated_answer = None

        return self.encoder.encode(self.question).detach().numpy(), reward, done, generated_answer

    # TODO - this should return a batch instead of a single observation
    def reset(self, *, seed=None, options=None):
        samples = self.dataset.data.sample(self.config.num_episodes_in_batch)

        # Format the questions and store them
        self.questions = "Question: " + samples["question"] + "\nAnswer: "
        self.questions = self.questions.tolist()
        self.ground_truths = samples["answer"].tolist()

        # Batch encoding of questions
        batched_observations = self.encoder.encode(self.questions).detach().numpy()

        return batched_observations

    # TODO - try running heavier model on colab and slurm
    def evaluate_prompt(self):
        generated_answer = self.llm.generate_answer(self.question)
        logger.debug(f"\nPrompt:\n{self.question}\n"
                     f"Generated answer:\n{generated_answer}\n"
                     f"Ground truth:\n{self.ground_truth}\n")

        reward = self.dataset.score(self.ground_truth, generated_answer)

        return reward, generated_answer
