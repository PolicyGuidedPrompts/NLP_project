import logging
import numpy as np

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
    def __init__(self, dataset, llm, retriever, seed, terminate_action=0):
        super(Environment, self).__init__()
        self.dataset = dataset
        self.retriever = retriever
        self.llm = llm
        self.terminate_action = terminate_action

        self.action_space = self.retriever.top_k + 1  # +1 for terminate action

        # Define observation space based on a sample observation
        # Half for generated prompt and half for storing the original question encodings
        self.observation_space = self.retriever.encode("Sample question for shape determination").shape[0] * 2

        self.seed = seed
        logger.info(f"Environment initialized with: "
                    f"{self.seed=}, "
                    f"{self.action_space=}, "
                    f"{self.observation_space=}, "
                    f"{self.llm.model_name=}, "
                    f"{self.retriever.model_name=}")

    # TODO - change self.question to prompt
    def step(self, action):
        if action == self.terminate_action:
            done = True
        else:
            index_given_action = self.top_k_closest_questions_indices[action-1]
            self.context_prompt = self.dataset.update_prompt(index_given_action, self.context_prompt)
            done = self.llm.is_prompt_too_long(f'{self.context_prompt}{self.question}')

        if done:
            reward, generated_answer = self.evaluate_prompt()
        else:
            reward = 0
            generated_answer = None

        prompt_encodings = self.retriever.encode(self.context_prompt)
        return np.concatenate([prompt_encodings, self.question_encodings]), reward, done, generated_answer

    def reset(self):
        self.question, self.ground_truth = self.dataset.reset()
        self.question_encodings = self.retriever.encode(self.question)
        # Adding Answer: to original question
        self.question = f'{self.question}\nAnswer: '
        self.context_prompt = ''
        self.prompt_encodings = self.retriever.encode(self.context_prompt)
        # Get retriever tok_k for question
        self.top_k_closest_questions_indices = self.retriever.retrieve(self.question_encodings)
        return np.concatenate([self.prompt_encodings, self.question_encodings])

    # TODO - hyper parameters as well as n_layers and heavier models try on slurm
    def evaluate_prompt(self):
        prompt = f'{self.context_prompt}{self.question}'
        generated_answer = self.llm.generate_answer(prompt)
        logger.debug(f"\nPrompt:\n{prompt}\n"
                     f"Generated answer:\n{generated_answer}\n"
                     f"Ground truth:\n{self.ground_truth}\n")

        reward = self.dataset.score(self.ground_truth, generated_answer)

        return reward, generated_answer
