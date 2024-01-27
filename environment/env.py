import logging
import numpy as np

logger = logging.getLogger("root")


class Environment:
    def __init__(self, dataset, llm, retriever, seed, terminate_action=0):
        super(Environment, self).__init__()
        self.dataset = dataset
        self.retriever = retriever
        self.llm = llm
        self.terminate_action = -1

        self.action_space = (
            self.retriever.top_k
            if hasattr(self.retriever, "top_k")
            else len(dataset.train_data)
        )

        # Define observation space based on a sample observation
        # Half for generated prompt and half for storing the original question encodings
        self.observation_space = (
            self.retriever.encode("Sample question for shape determination").shape[0]
            * 2
        )

        self.seed = seed
        logger.info(
            f"Environment initialized with: "
            f"{self.seed=}, "
            f"{self.action_space=}, "
            f"{self.observation_space=}, "
            f"{self.llm.model_name=}, "
            f"{self.retriever.model_name=}"
        )

    def step(self, action, done):
        index_given_action = self.top_k_closest_questions_indices[action]
        self.context_prompt = self.dataset.update_prompt(index_given_action, self.context_prompt)

        if done:
            reward = self.evaluate_prompt()
        else:
            reward = 0.0

        prompt_encodings = self.retriever.encode(self.context_prompt)
        return np.concatenate([prompt_encodings, self.question_encodings]), reward, done

    def reset(self, mode="train", index=None):
        self.question, self.initial_prompt, self.ground_truth = self.dataset.reset(
            mode=mode, index=index
        )
        self.question_encodings = self.retriever.encode(self.question)
        self.context_prompt = ""
        self.prompt_encodings = self.retriever.encode(self.context_prompt)
        self.top_k_closest_questions_indices = self.retriever.retrieve(
            encoding=self.question_encodings, mode=mode
        )
        return np.concatenate([self.prompt_encodings, self.question_encodings])

    def evaluate_prompt(self):
        prompt = (
            f"{self.dataset.prompt_prefix}{self.context_prompt}{self.initial_prompt}"
        )
        generated_answer = self.llm.generate_answer(prompt)
        logger.info(
            f"\nPrompt:\n{prompt}\n"
            f"Generated answer:\n{generated_answer}\n"
            f"Ground truth:\n{self.ground_truth}\n"
        )

        reward = self.dataset.score(self.ground_truth, generated_answer)

        return reward
