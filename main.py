import torch
# import wandb
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT3LMHeadModel
# from torch.distributions import Categorical
from datasets import load_dataset

from utils.utils import load_or_download_model, load_or_download_llm_model
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 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 (llama 2 read acces token)

# Dataset path
DATA_PATH = "path_to_your_dataset"


# Initialize wandb
# wandb.init(project="NLP-RL")


class Environment:
    def __init__(self, training_dataset, special_action=0):
        self.training_dataset = training_dataset  # A list of question+answer pairs
        self.encoder_tokenizer, self.encoder_model = load_or_download_model()
        self.llm_tokenizer, self.llm_model = load_or_download_llm_model()
        self.special_action = special_action
        self.reset()

    def step(self, action):
        done = False
        reward = 0

        if action == self.special_action:
            done = True
            reward = self.evaluate_prompt()
        else:
            # Concatenate the selected question and answer to the current prompt
            sampled_question, sampled_answer = self.training_dataset.iloc[action]
            self.question = f"{sampled_question}\n{sampled_answer}\n{self.question}"

        return self.encode_question(self.question), reward, done

    def reset(self):
        # Sample a new question and answer from the training dataset
        sample = self.training_dataset.sample(1).iloc[0]
        self.question, self.answer = sample["question"]+'\n', str(sample["answer"])
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
        inputs = self.llm_tokenizer.encode(self.question, return_tensors="pt")
        with torch.no_grad():
            outputs = self.llm_model.generate(inputs, max_length=150, temperature=0.7)
        generated_answer = self.llm_tokenizer.decode(
            outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True
        )
        # Compare the generated answer to the correct answer
        if generated_answer == self.answer:
            return 1
        else:
            return -1


class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class Agent:
    def __init__(self, state_size, action_size, lr=0.01, start_epsilon=1.0, end_epsilon=0.001, decay_steps=1e8):
        self.policy_net = PolicyNet(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.steps_done = 0

    def get_epsilon(self):
        fraction_of_steps = min(self.steps_done / self.decay_steps, 1)
        return self.start_epsilon + fraction_of_steps * (self.end_epsilon - self.start_epsilon)

    def get_action(self, state):
        self.steps_done += 1
        epsilon = self.get_epsilon()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = self.policy_net(state)
        if np.random.rand() < epsilon:
            action = torch.randint(high=action_size, size=(1,))  # Random action
        else:
            action = torch.argmax(action_probs)  # Best action
        return action.item()

    def train(self, env, num_episodes, episodes_per_update):
        for episode in range(num_episodes):
            all_states = []
            all_actions = []
            all_returns = []

            for _ in range(episodes_per_update):
                states, actions, rewards = self.collect_episode(env)
                returns = [sum(rewards[i:]) for i in range(len(rewards))]
                all_states.extend(states)
                all_actions.extend(actions)
                all_returns.extend(returns)

            self.optimizer.zero_grad()

            for s, a, G in zip(all_states, all_actions, all_returns):
                action_probs = self.policy_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
                log_prob = torch.log(action_probs[0, a])
                loss = -log_prob * G
                loss.backward()

            self.optimizer.step()

    def collect_episode(self, env):
        state = env.reset()
        done = False
        states = []
        actions = []
        rewards = []
        episode_len = 0  # TODO - remove this

        while not done:
            episode_len += 1  # TODO - remove this
            if episode_len > 10:
                action = 0
            else:
                action = self.get_action(state)
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        return states, actions, rewards


if __name__ == "__main__":
    # Define the environment and the agent
    # Train the agent
    # Evaluate the agent
    dataset = load_dataset("wics/strategy-qa")["test"].to_pandas()[
        ["question", "answer"]
    ]
    question, answer = dataset.iloc[0]
    env = Environment(
        training_dataset=dataset,
    )

    state_size = 768
    action_size = 2290

    num_episodes = 1000
    episodes_per_update = 10

    agent = Agent(state_size, action_size)
    agent.train(env, num_episodes, episodes_per_update)

    print("stop")
