# import torch
# import wandb
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT3LMHeadModel
# from torch.distributions import Categorical
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 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 (llama 2 read acces token)
import os

# Dataset path
DATA_PATH = "path_to_your_dataset"

# Initialize wandb
# wandb.init(project="NLP-RL")

from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM


class Environment:
    def __init__(
        self, prompt, correct_answer, gpt_model_name, special_action, training_dataset
    ):
        model_dir = os.path.abspath("./saved_model")

        # Try loading the model from a local directory. If it doesn't exist, download it and save it locally
        if (
            os.path.exists(os.path.join(model_dir, "config.json"))
            and os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
            and os.path.exists(os.path.join(model_dir, "tokenizer_config.json"))
            and os.path.exists(os.path.join(model_dir, "vocab.json"))
        ):
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
            self.model = GPT2LMHeadModel.from_pretrained(gpt_model_name)
            self.tokenizer.save_pretrained(model_dir)
            self.model.save_pretrained(model_dir)

        self.original_question = prompt
        self.correct_answer = correct_answer
        self.special_action = special_action
        self.training_dataset = training_dataset  # A list of question+answer pairs
        self.reset()

    def step(self, action):
        done = False
        reward = 0

        if action == self.special_action:
            done = True
            reward = self.evaluate_prompt()
        else:
            # Concatenate the selected question and answer to the current prompt
            question,answer = self.training_dataset.iloc[action]
            self.current_prompt = f"{question}\n{answer}\n{self.current_prompt}"

        return self.current_prompt, reward, done

    def reset(self):
        self.current_prompt = self.original_question

    def render(self):
        print(self.current_prompt)

    def evaluate_prompt(self):
        # Generate an answer from the GPT model for the current prompt
        inputs = self.tokenizer.encode(self.current_prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=150, temperature=0.7)
        generated_answer = self.tokenizer.decode(
            outputs[:, inputs.shape[-1] :][0], skip_special_tokens=True
        )

        # Compare the generated answer to the correct answer
        if generated_answer == self.correct_answer:
            return 1
        else:
            return -1


class Agent:
    """
    def __init__(self, encoding_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(encoding_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(encoding_model_name)

    def get_action(self, state):
        # Given a state, select an action
        # The action is selecting which question+answer to add to the prompt
        pass
        """


def train():
    """
    # Load the dataset
    # For each episode:
        # Get initial state (question)
        # While not done:
            # Agent chooses an action based on the current state
            # Update the state based on the action
            # If the action was the special action:
                # Get the reward from the environment
                # Update the agent based on the reward
    pass
    """


def evaluate():
    # Similar to the train function, but without updating the agent
    pass


def collect_episode(env, policy_net):
    """
    states = []
    actions = []
    rewards = []

    s = env.reset()  # Reset environment to initial state
    done = False

    while not done:
        states.append(s)

        action_probs = policy_net(torch.tensor(s, dtype=torch.float32))  # Get action probabilities from policy network
        a = np.random.choice(len(action_probs), p=action_probs.detach().numpy())  # Sample action

        actions.append(a)

        s, r, done, _ = env.step(a)  # Take action in environment

        rewards.append(r)

    return states, actions, rewards
    """


# Usage:


def reinforce():
    """
    # Assume these are lists that you've collected from an episode
    states, actions, rewards = collect_episode(env, policy_net)

    # Assume policy_net is your policy network
    optimizer = torch.optim.Adam(policy_net.parameters())

    optimizer.zero_grad()

    for s_t, a_t, G_t in zip(states, actions, returns):
        action_probs = policy_net(s_t)  # forward pass through the network
        log_prob = torch.log(action_probs[a_t])  # log probability of the action taken
        loss = -log_prob * G_t  # negative because we want to do gradient ascent
        loss.backward()  # accumulate gradients

    optimizer.step()  # perform a gradient ascent step
    """


if __name__ == "__main__":
    # Define the environment and the agent
    # Train the agent
    # Evaluate the agent
    dataset = load_dataset("wics/strategy-qa")["test"].to_pandas()[
        ["question", "answer"]
    ]
    question, answer = dataset.iloc[0]
    env = Environment(
        prompt=question,
        correct_answer=answer,
        gpt_model_name="gpt2",
        special_action=0,
        training_dataset=dataset,
    )
    a, b, c = env.step(1)
    a, b, c = env.step(2)
    a, b, c = env.step(3)
    a_tag, b_tag, c_tag = env.step(0)
    print("stop")
