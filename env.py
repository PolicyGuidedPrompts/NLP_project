import torch
from utils.utils import load_or_download_model, load_or_download_llm_model
import gym
from gym.spaces import Discrete, Box

_NUMBER_OF_SPECIAL_ACTIONS = 1


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


    # TODO - punish on long episodes
    # TODO- can make this more readable
    # TODO - if time permits can remove last question encoding
    def step(self, action, reached_max_episode_length=False):
        stop = reached_max_episode_length or action == self.special_action
        done = False
        reward = 0
        generated_answer = None

        if stop:
            done = True
            reward, generated_answer = self.evaluate_prompt()
        else:
            # Concatenate the selected question and answer to the current prompt
            sampled_question, sampled_answer = self.training_dataset.iloc[action]
            self.question = f"Question: {sampled_question}\nAnswer: {sampled_answer}\n{self.question}"

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

    def evaluate_prompt(self):
        # TODO - maybe need to revisit this one

        # Generate an answer from the BERT model for the current prompt
        print(f"Prompt:\n{self.question}\n")
        inputs = self.llm_tokenizer.encode(self.question, return_tensors="pt")
        with torch.no_grad():
            outputs = self.llm_model.generate(inputs, max_length=250, temperature=0.7)
        generated_answer = self.llm_tokenizer.decode(
            outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True
        )
        print(f"Generated answer:\n{generated_answer}\n")
        # Compare the generated answer to the correct answer
        if generated_answer == self.answer:
            return 1, generated_answer
        else:
            return -1, generated_answer

    # TODO - maybe need to implement this one
    def close(self):
        # This method can be used to perform any cleanup when the environment is closed.
        pass
