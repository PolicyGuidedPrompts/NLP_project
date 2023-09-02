import logging

# TODO - Deberta tokenizer and model

# TODO - speak with Nachum about Masters partition slurm

# TODO - punish on long episodes

# TODO
# Roi suggestion
# choosing an action will actually choose a random vector
# will then choose the closest vector to that random vector

# Questions:
# Ask about budget

logger = logging.getLogger('root')


class Environment:
    def __init__(self, dataset, llm, encoder, seed, terminate_action=0):
        super(Environment, self).__init__()
        self.dataset = dataset
        self.encoder = encoder
        self.llm = llm
        self.terminate_action = terminate_action

        self.action_space = len(self.dataset.data) + 1  # +1 for terminate action

        # Define observation space based on a sample observation
        sample_observation = self.encoder.encode("Sample question for shape determination").numpy()
        self.observation_space = sample_observation

        self.seed = seed
        self.reset()
        logger.info(f"Environment initialized with: "
                    f"({self.seed=}, "
                    f"{self.action_space=}, "
                    f"{self.observation_space=}, "
                    f"{self.llm.model_name=}, "
                    f"{self.encoder.model_name=})")

    def _update_prompt_based_on_action(self, action):
        sampled_question, sampled_answer = self.dataset.data.iloc[action]
        self.question = f"Question: {sampled_question}\nAnswer: {sampled_answer}\n{self.question}"

    def step(self, action):
        if action == self.terminate_action:
            done = True
        else:
            self._update_prompt_based_on_action(action)
            done = self.llm.is_prompt_too_long(self.question)

        if done:
            reward, generated_answer = self.evaluate_prompt()
        else:
            reward = 0
            generated_answer = None

        return self.encoder.encode(self.question).detach().numpy(), reward, done, generated_answer

    def reset(self, *, seed=None, options=None):
        sample = self.dataset.data.sample(1).iloc[0]
        self.question, self.ground_truth = f'Question: {sample["question"]}\nAnswer: ', sample["answer"]
        return self.encoder.encode(self.question).detach().numpy()

    # TODO - hyper parameters as well as n_layers and heavier models try on slurm
    def evaluate_prompt(self):
        generated_answer = self.llm.generate_answer(self.question)
        logger.debug(f"\nPrompt:\n{self.question}\n"
                     f"Generated answer:\n{generated_answer}\n"
                     f"Ground truth:\n{self.ground_truth}\n")

        reward = self.dataset.score(self.ground_truth, generated_answer)

        return reward, generated_answer
