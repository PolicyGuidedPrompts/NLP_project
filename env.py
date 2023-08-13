import torch
from utils.utils import load_or_download_model, load_or_download_llm_model
import gym
from gym.spaces import Discrete, Box
import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

_NUMBER_OF_SPECIAL_ACTIONS = 1

MAX_BUFFER = 200


class Environment(gym.Env):
    # TODO - training_dataset + llm should be defined from configuration file
    def __init__(self, training_dataset, special_action=0):
        super(Environment, self).__init__()
        self.training_dataset = training_dataset
        self.encoder_tokenizer, self.encoder_model = load_or_download_model()
        self.llm_tokenizer, self.llm_model = load_or_download_llm_model()
        self.special_action = special_action

        # Define action space
        self.action_space = Discrete(len(self.training_dataset) + _NUMBER_OF_SPECIAL_ACTIONS)

        # Define observation space based on a sample observation
        sample_observation = self.encode_question("Sample question for shape determination")
        sample_observation_np = sample_observation.numpy()
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=sample_observation_np.shape,
                                     dtype=sample_observation_np.dtype)

        self.reset()

    # TODO - not sure should be a part of env class
    def get_tokenized_length(self, question):
        tokenized = self.encoder_tokenizer(question, return_tensors="pt", truncation=True, padding=True)
        return tokenized['input_ids'].shape[1]

    # TODO - punish on long episodes
    def step(self, action):
        if action == self.special_action:
            done = True
        else:
            # Concatenate the selected question and answer to the current prompt
            sampled_question, sampled_answer = self.training_dataset.iloc[action]
            self.question = f"Question: {sampled_question}\nAnswer: {sampled_answer}\n{self.question}"
            # TODO - MAX_BUFFER will be determined by the model
            done = self.get_tokenized_length(self.question) > MAX_BUFFER

        if done:
            reward, generated_answer = self.evaluate_prompt()
        else:
            reward = 0
            generated_answer = None

        return self.encode_question(self.question), reward, done, generated_answer

    def seed(self, seed):
        self.seed = seed

    def reset(self):
        # Sample a new question and answer from the training dataset
        sample = self.training_dataset.sample(1).iloc[0]
        self.question, self.answer = f'Question: {sample["question"]}\nAnswer: ', sample["answer"]
        return self.encode_question(self.question)

    def render(self):
        print(self.question)

    def encode_question(self, question):
        # Encode the question with the BERT tokenizer and model
        inputs = self.encoder_tokenizer(question, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.encoder_model(**inputs)
        # Use the last hidden state as the question representation
        return outputs.last_hidden_state[0, 0, :]

    # TODO - try running heavier model on colab and slurm
    def evaluate_prompt(self):
        # Use GPT-3.5-turbo to generate an answer for the current prompt
        print(f"Prompt:\n{self.question}\n")

        # Making an API call using the chat endpoint
        # TODO - model should be configurable
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": self.question}
            ],
            # TODO - 15 for now, remove this later, should be configurable
            max_tokens=15  # Limit the output to 15 tokens
        )

        generated_answer = response['choices'][0]['message']['content'].strip()
        print(f"Generated answer:\n{generated_answer}\n")
        print(f"Ground truth:\n{self.answer}\n")

        # Compare the generated answer to the correct answer
        if generated_answer == self.answer:
            return 1, generated_answer
        else:
            return -1, generated_answer

    # TODO - maybe need to implement this one
    def close(self):
        # This method can be used to perform any cleanup when the environment is closed.
        pass
