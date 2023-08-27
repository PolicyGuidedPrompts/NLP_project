import os

import gym
import openai
import torch
from gym.spaces import Discrete, Box

openai.api_key = os.environ.get('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


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

class Environment(gym.Env):

    def __init__(self, dataset, llm, encoder, seed, terminate_action=0):
        super(Environment, self).__init__()
        self.dataset = dataset
        self.encoder = encoder
        self.llm = llm
        self.terminate_action = terminate_action

        # Define action space
        self.action_space = Discrete(len(self.dataset) + 1)  # +1 for terminate action

        # Define observation space based on a sample observation
        sample_observation = self.encoder.encode("Sample question for shape determination").numpy()
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=sample_observation.shape,
                                     dtype=sample_observation.dtype)

        self.seed = seed
        self.reset()

    def _update_prompt_based_on_action(self, action):
        sampled_question, sampled_answer = self.dataset.iloc[action]
        self.question = f"Question: {sampled_question}\nAnswer: {sampled_answer}\n{self.question}"

    # TODO - should be based on the llm tokenizer (note that GPT3.5 doesn't have tokenizer)
    def _is_prompt_too_long(self):
        tokenized = self.encoder.tokenizer(self.question, return_tensors="pt", truncation=True, padding=True)
        tokenized_len = tokenized['input_ids'].shape[1]
        return tokenized_len > self.llm.max_prompt_tokenized_len

    def step(self, action):
        if action == self.terminate_action:
            done = True
        else:
            self._update_prompt_based_on_action(action)
            done = self._is_prompt_too_long()

        if done:
            reward, generated_answer = self.evaluate_prompt()
        else:
            reward = 0
            generated_answer = None

        return self.encoder.encode(self.question).detach().numpy(), reward, done, generated_answer

    # TODO - this should return a batch instead of a single observation
    def reset(self, *, seed=None, options=None):
        # Sample a new question and answer from the training dataset
        sample = self.dataset.sample(1).iloc[0]
        self.question, self.ground_truth = f'Question: {sample["question"]}\nAnswer: ', sample["answer"]
        return self.encoder.encode(self.question).detach().numpy()

    # TODO - implement this per defined config metric
    def score_generated_answer(self, generated_answer):
        return NotImplemented

    # TODO - try running heavier model on colab and slurm
    # TODO - remove print, log instead using decorators
    def evaluate_prompt(self):
        print(f"Prompt:\n{self.question}\n")
        generated_answer = self.llm.generate_answer(self.question)
        print(f"Generated answer:\n{generated_answer}\n")
        print(f"Ground truth:\n{self.ground_truth}\n")

        # TODO - based on configured reward metric, mainly determined by the dataset
        # Compare the generated answer to the correct answer
        if generated_answer == self.ground_truth:
            return 1, generated_answer
        else:
            return -1, generated_answer
