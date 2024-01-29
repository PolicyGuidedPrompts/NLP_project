import logging
import os
import re
import string
from abc import ABC, abstractmethod
from collections import Counter
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from datasets import load_dataset

logger = logging.getLogger("root")


class Dataset(ABC):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    datasets_dir = os.path.join(script_dir, "../saved_datasets")
    task_name = ''

    prompt_prefix = "See the questions and answers below and answer the last question in the same fashion.\n"

    def __init__(self):
        self.train_data, self.test_data = self.load_from_repository()
        logger.info(f"Scoring method: {self.get_scoring_method_name()}")

    def score(self, ground_truth, generated_answer):
        return self._f1_score(ground_truth, generated_answer)

    def _f1_score(self, ground_truth, generated_answer):
        try:
            prediction_tokens = self.normalize_answer(generated_answer).split()
            ground_truth_tokens = self.normalize_answer(ground_truth).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0.0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2.0 * precision * recall) / (precision + recall)
            return 2.0 * f1 - 1.0  # normalize between -1 and 1
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return -1.0

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def get_scoring_method_name():
        return "f1_score"

    @property
    def dataset_path(self):
        if hasattr(self, "repository"):
            return self.repository + "/" + self.dataset_name
        return self.dataset_name

    def prepare_dataset_to_retriever(self):
        return self.train_data["question"].to_numpy()

    @abstractmethod
    def load_from_repository(self):
        pass

    @abstractmethod
    def reset(self, mode, index):
        pass

    @abstractmethod
    def update_prompt(self, action, current_prompt):
        pass


class StrategyQaDataset(Dataset):
    repository = "wics"
    dataset_name = "strategy-qa"

    prompt_prefix = (
        "See the questions and answers below and "
        "answer the last question in the same fashion base on the facts.\n"
    )

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name} from {self.repository}")
        df = (
            load_dataset(self.dataset_path, cache_dir=self.datasets_dir)["test"]
            .to_pandas()[["question", "answer", "facts"]]
            .assign(answer=lambda x: x["answer"].astype(str))
        )
        df["facts"] = (
            df["facts"].astype(str).apply(lambda x: "".join(ast.literal_eval(x)))
        )

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        return df_train, df_test

    def reset(self, mode, index):
        if mode == "train":
            sample = self.train_data.sample(1).iloc[0]
        else:
            sample = self.test_data.iloc[index]
        question, ground_truth = sample["question"], sample["answer"]
        initial_prompt = (
            f'Question: {sample["question"]}\n'
            f'Facts: {sample["facts"]}\n'
            f"Answer: "
        )
        return question, initial_prompt, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.train_data.iloc[action]
        new_prompt = (
            f'Question: {sample["question"]}\n'
            f'Facts: {sample["facts"]}\n'
            f'Answer: {sample["answer"]}\n'
            f"{current_prompt}"
        )
        return new_prompt

    def score(self, ground_truth, generated_answer):
        try:
            return 1.0 if generated_answer.lower() == ground_truth.lower() else -1.0
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return -1.0

    @staticmethod
    def get_scoring_method_name():
        return "exact-match"


class SquadDataset(Dataset):
    dataset_name = "squad"

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name}")
        data = load_dataset(self.dataset_path, cache_dir=self.datasets_dir)

        df_train = data["train"].to_pandas()
        df_train["answers"] = df_train["answers"].apply(
            lambda x: x["text"] if x else None
        )

        df_test = data["validation"].to_pandas()
        df_test["answers"] = df_test["answers"].apply(
            lambda x: x["text"] if x else None
        )

        return (
            df_train[["question", "answers", "context"]],
            df_test[["question", "answers", "context"]],
        )

    def reset(self, mode, index):
        if mode == "train":
            sample = self.train_data.sample(1).iloc[0]
        else:
            sample = self.test_data.iloc[index]
        question, ground_truth = sample["question"], sample["answers"]
        initial_prompt = (
            f'Question: {sample["question"]}\n'
            f'Context: {sample["context"]}\n'
            f"Answer: "
        )
        return question, initial_prompt, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.train_data.iloc[action]
        new_prompt = (
            f'Question: {sample["question"]}\n'
            f'Context: {sample["context"]}\n'
            f'Answer: {sample["answers"][0]}\n'
            f"{current_prompt}"
        )
        return new_prompt

    def score(self, ground_truths, generated_answer):
        try:
            return max(
                [
                    self._f1_score(ground_truth, generated_answer)
                    for ground_truth in ground_truths
                ]
            )
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return -1.0

    @staticmethod
    def get_scoring_method_name():
        return "custom f1_score"


class OpenTDB(Dataset):
    dataset_name = "open_tdb"
    local_path_to_data_set = "../data/open_tdb.csv"

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name} from local CSV")
        data = pd.read_csv(os.path.join(self.script_dir, self.local_path_to_data_set))
        df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
        return df_train, df_test

    def reset(self, mode, index):
        if mode == "train":
            sample = self.train_data.sample(1).iloc[0]
        else:
            sample = self.test_data.iloc[index]
        question, ground_truth = sample["question"], sample["answer"]
        initial_prompt = f'Question: {sample["question"]}\n' f"Answer: "
        return question, initial_prompt, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.train_data.iloc[action]
        new_prompt = (
            f'Question: {sample["question"]}\n'
            f'Answer: {sample["answer"]}\n'
            f"{current_prompt}"
        )
        return new_prompt


class PAWSDataset(Dataset):
    dataset_name = "paws"

    prompt_prefix = (
        "Below are sentences which one sentence can be a paraphrase of the other. "
        "Answer the last two sentences with resect to given examples.\n"
    )

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name}")
        data = load_dataset(self.dataset_path, "labeled_final", cache_dir=self.datasets_dir)

        df_train = data["train"].to_pandas()
        df_test = data["test"].to_pandas()

        label_map = {0: 'No', 1: 'Yes'}
        df_train['label'] = df_train['label'].map(label_map)
        df_test['label'] = df_test['label'].map(label_map)

        return df_train, df_test

    def reset(self, mode, index):
        if mode == "train":
            sample = self.train_data.sample(1).iloc[0]
        else:
            sample = self.test_data.iloc[index]
        question, ground_truth = f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}", sample[
            "label"]
        initial_prompt = f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}\nIs Paraphrase? "
        return question, initial_prompt, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.train_data.iloc[action]
        new_prompt = (
            f"Sentence 1: {sample['sentence1']}\n"
            f"Sentence 2: {sample['sentence2']}\n"
            f"Is Paraphrase? {sample['label']}\n"
            f"{current_prompt}"
        )
        return new_prompt

    def prepare_dataset_to_retriever(self):
        retriever_data = self.train_data.apply(
            lambda row: f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}", axis=1)
        return retriever_data.to_numpy()

    def score(self, ground_truth, generated_answer):
        try:
            return 1.0 if generated_answer.lower().strip() == ground_truth.lower().strip() else -1.0
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return -1.0


class GlueMNLIDataset(Dataset):
    dataset_name = "glue"
    task_name = "mnli"

    prompt_prefix = (
        "Below are sentences which one sentence can entail/neutral/contradicts the other. "
        "Answer the last two sentences with resect to given examples.\n"
    )

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name} for task {self.task_name}")
        data = load_dataset(self.dataset_path, self.task_name, cache_dir=self.datasets_dir)

        df_train = data["train"].to_pandas()
        df_test = data["test_matched"].to_pandas()

        label_map = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}
        df_train['label'] = df_train['label'].map(label_map)
        df_test['label'] = df_test['label'].map(label_map)

        return df_train, df_test

    def reset(self, mode, index):
        if mode == "train":
            sample = self.train_data.sample(1).iloc[0]
        else:
            sample = self.test_data.iloc[index]
        question, ground_truth = f"Premise: {sample['premise']}\nHypothesis: {sample['hypothesis']}", sample["label"]
        initial_prompt = f"Premise: {sample['premise']}\nHypothesis: {sample['hypothesis']}\nNLI Verdict: "
        return question, initial_prompt, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.train_data.iloc[action]
        new_prompt = (
            f"Premise: {sample['premise']}\n"
            f"Hypothesis: {sample['hypothesis']}\n"
            f"NLI Verdict: {sample['label']}\n"
            f"{current_prompt}"
        )
        return new_prompt

    def prepare_dataset_to_retriever(self):
        retriever_data = self.train_data.apply(
            lambda row: f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}", axis=1)
        return retriever_data.to_numpy()

    def score(self, ground_truth, generated_answer):
        try:
            return 1.0 if generated_answer.lower().strip() == ground_truth.lower().strip() else -1.0
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return -1.0


class GlueRTEDataset(Dataset):
    dataset_name = "glue"
    task_name = "rte"

    prompt_prefix = (
        "Below are pairs of sentences. Determine if the hypothesis is entailed by the premise. "
        "Answer with either Not Entailment/Entailment. "
        "Answer the last pair of sentences with respect to the given examples.\n"
    )

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name} for task {self.task_name}")
        data = load_dataset(self.dataset_path, self.task_name, cache_dir=self.datasets_dir)

        df_train = data["train"].to_pandas()
        df_test = data["test"].to_pandas()

        label_map = {1: 'Not Entailment', 0: 'Entailment'}
        df_train['label'] = df_train['label'].map(label_map)
        df_test['label'] = df_test['label'].map(label_map)

        return df_train, df_test

    def reset(self, mode, index):
        if mode == "train":
            sample = self.train_data.sample(1).iloc[0]
        else:
            sample = self.test_data.iloc[index]
        question, ground_truth = f"Premise: {sample['sentence1']}\nHypothesis: {sample['sentence2']}", sample["label"]
        initial_prompt = f"Premise: {sample['sentence1']}\nHypothesis: {sample['sentence2']}\nNLI Verdict: "
        return question, initial_prompt, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.train_data.iloc[action]
        new_prompt = (
            f"Premise: {sample['sentence1']}\n"
            f"Hypothesis: {sample['sentence2']}\n"
            f"NLI Verdict: {sample['label']}\n"
            f"{current_prompt}"
        )
        return new_prompt

    def prepare_dataset_to_retriever(self):
        retriever_data = self.train_data.apply(
            lambda row: f"Premise: {row['sentence1']}\nHypothesis: {row['sentence2']}", axis=1)
        return retriever_data.to_numpy()

    def score(self, ground_truth, generated_answer):
        try:
            return 1.0 if generated_answer.lower().strip() == ground_truth.lower().strip() else -1.0
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return -1.0


class GlueCoLADataset(Dataset):
    dataset_name = "glue"
    task_name = "cola"

    prompt_prefix = (
        "Below are sentences. Determine if each sentence is grammatically acceptable. "
        "Answer with either Acceptable/Unacceptable. "
        "Answer the last sentence with respect to the given examples.\n"
    )

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name} for task {self.task_name}")
        data = load_dataset(self.dataset_path, self.task_name, cache_dir=self.datasets_dir)

        df_train = data["train"].to_pandas()
        df_test = data["test"].to_pandas()

        label_map = {1: 'Acceptable', 0: 'Unacceptable'}
        df_train['label'] = df_train['label'].map(label_map)
        df_test['label'] = df_test['label'].map(label_map)

        return df_train, df_test

    def reset(self, mode, index):
        if mode == "train":
            sample = self.train_data.sample(1).iloc[0]
        else:
            sample = self.test_data.iloc[index]
        question, ground_truth = sample['sentence'], sample["label"]
        initial_prompt = f"Sentence: {sample['sentence']}\nGrammar Verdict: "
        return question, initial_prompt, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.train_data.iloc[action]
        new_prompt = (
            f"Sentence: {sample['sentence']}\n"
            f"Grammar Verdict: {sample['label']}\n"
            f"{current_prompt}"
        )
        return new_prompt

    def prepare_dataset_to_retriever(self):
        retriever_data = self.train_data['sentence'].to_numpy()
        return retriever_data

    def score(self, ground_truth, generated_answer):
        try:
            return 1.0 if generated_answer.lower().strip() == ground_truth.lower().strip() else -1.0
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return -1.0


class GlueMRPCDataset(Dataset):
    dataset_name = "glue"
    task_name = "mrpc"

    prompt_prefix = (
        "Below are pairs of sentences. Determine if the sentences are equivalent in meaning. "
        "Answer with either Equivalent/Not Equivalent. "
        "Answer the last pair of sentences with respect to the given examples.\n"
    )

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name} for task {self.task_name}")
        data = load_dataset(self.dataset_path, self.task_name, cache_dir=self.datasets_dir)

        df_train = data["train"].to_pandas()
        df_test = data["test"].to_pandas()

        label_map = {1: 'Equivalent', 0: 'Not Equivalent'}
        df_train['label'] = df_train['label'].map(label_map)
        df_test['label'] = df_test['label'].map(label_map)

        return df_train, df_test

    def reset(self, mode, index):
        if mode == "train":
            sample = self.train_data.sample(1).iloc[0]
        else:
            sample = self.test_data.iloc[index]
        question = f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}"
        ground_truth = sample["label"]
        initial_prompt = f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}\nParaphrase Verdict: "
        return question, initial_prompt, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.train_data.iloc[action]
        new_prompt = (
            f"Sentence 1: {sample['sentence1']}\n"
            f"Sentence 2: {sample['sentence2']}\n"
            f"Paraphrase Verdict: {sample['label']}\n"
            f"{current_prompt}"
        )
        return new_prompt

    def prepare_dataset_to_retriever(self):
        retriever_data = self.train_data.apply(
            lambda row: f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}", axis=1)
        return retriever_data.to_numpy()

    def score(self, ground_truth, generated_answer):
        try:
            return 1.0 if generated_answer.lower().strip() == ground_truth.lower().strip() else -1.0
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return -1.0


class QuoraDataset(Dataset):
    dataset_name = "quora"

    prompt_prefix = (
        "Below are pairs of questions. Determine if the questions are duplicates of each other. "
        "Answer with either Duplicate/Not Duplicate. "
        "Answer the last pair of questions with respect to the given examples.\n"
    )

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name}")
        data = load_dataset(self.dataset_path, cache_dir=self.datasets_dir)

        df_train = data["train"].to_pandas()

        df_train['question1'] = df_train['questions'].apply(lambda x: x['text'][0])
        df_train['question2'] = df_train['questions'].apply(lambda x: x['text'][1])

        label_map = {False: 'Not Duplicate', True: 'Duplicate'}
        df_train['is_duplicate'] = df_train['is_duplicate'].map(label_map)

        return df_train, []

    def reset(self, mode, index):
        if mode == "train":
            sample = self.train_data.sample(1).iloc[0]
        else:
            sample = self.test_data.iloc[index]
        question = f"Question 1: {sample['question1']}\nQuestion 2: {sample['question2']}"
        ground_truth = sample["is_duplicate"]
        initial_prompt = f"Question 1: {sample['question1']}\nQuestion 2: {sample['question2']}\nDuplicate Verdict: "
        return question, initial_prompt, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.train_data.iloc[action]
        new_prompt = (
            f"Question 1: {sample['question1']}\n"
            f"Question 2: {sample['question2']}\n"
            f"Duplicate Verdict: {sample['is_duplicate']}\n"
            f"{current_prompt}"
        )
        return new_prompt

    def prepare_dataset_to_retriever(self):
        retriever_data = self.train_data.apply(
            lambda row: f"Question 1: {row['question1']}\nQuestion 2: {row['question2']}", axis=1)
        return retriever_data.to_numpy()

    def score(self, ground_truth, generated_answer):
        try:
            return 1.0 if generated_answer.lower().strip() == ground_truth.lower().strip() else -1.0
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return -1.0


class AquaRat(Dataset):
    dataset_name = "aqua_rat"
    local_path_to_data_set = "../data/aqua_rat.csv"

    prompt_prefix = (
        "See the questions and answers below and answer the last question in the same fashion."
        "The final answer should begin with: 'The final answer is: A/B/C/D/E'\n"
    )

    def load_from_repository(self):
        logger.info(f"Loading dataset {self.dataset_name} from local CSV")
        df = pd.read_csv(os.path.join(self.script_dir, self.local_path_to_data_set))
        df["options"] = df["options"].apply(lambda x: "\n".join(ast.literal_eval(x)))
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        return df_train, df_test

    def reset(self, mode, index):
        if mode == "train":
            sample = self.train_data.sample(1).iloc[0]
        else:
            sample = self.test_data.iloc[index]
        question, ground_truth = sample["question"], sample["answer"]
        initial_prompt = (
            f'Question: {sample["question"]}\n' f'{sample["options"]}\n' f"Answer: "
        )
        return question, initial_prompt, ground_truth

    def update_prompt(self, action, current_prompt):
        sample = self.train_data.iloc[action]
        new_prompt = (
            f'Question: {sample["question"]}\n'
            f'{sample["options"]}\n'
            f'Answer: {sample["rationale"]}\n'
            f'The final answer is: {sample["answer"]}\n'
            f"{current_prompt}"
        )
        return new_prompt

    @staticmethod
    def extract_final_answer(generated_answer):
        match = re.search(r"The final answer is: (\w+)", generated_answer)
        return match.group(1) if match else None

    def score(self, ground_truth, generated_answer):
        try:
            if ground_truth.lower() == generated_answer.lower():
                return 1.0
            generated_answer = self.extract_final_answer(generated_answer)
            return 1.0 if generated_answer == ground_truth else -1.0
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            return -1.0

    @staticmethod
    def get_scoring_method_name():
        return "custom exact-match"


AVAILABLE_DATASETS = {
    "strategy-qa": StrategyQaDataset,
    "squad": SquadDataset,
    "open-tdb": OpenTDB,
    "aqua-rat": AquaRat,
    "paws": PAWSDataset,
    "glue-mnli": GlueMNLIDataset,
    "glue-rte": GlueRTEDataset,
    "glue-cola": GlueCoLADataset,
    "glue-mrpc": GlueMRPCDataset,
    "quora": QuoraDataset,
}


class DatasetFactory:
    @staticmethod
    def create_dataset(dataset_name):
        if dataset_name in AVAILABLE_DATASETS:
            return AVAILABLE_DATASETS[dataset_name]()
        else:
            logger.error(f"Dataset {dataset_name} not supported!")
            raise ValueError(f"Dataset {dataset_name} not supported!")
