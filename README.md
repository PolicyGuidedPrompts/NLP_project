### Project Overview

This project is a reinforcement learning-based approach to optimize the performance of a question-answering system. The system uses a retriever model to fetch relevant documents and a language model to generate answers based on the retrieved documents. The policy gradient method is employed to train the system, with the option to use the Proximal Policy Optimization (PPO) algorithm.

### Modules

1. [**Dataset**](#dataset-module)
2. [**LLM Model**](#llm-model-module)
3. [**Retriever Model**](#retriever-model-module)
4. [**Environment**](#environment-module)
5. [**Policy Search**](#policy-search-module)

### Dataset Module

The Dataset module is a crucial component of the project, responsible for managing and providing access to various datasets. 

#### Functionality:

- **Dataset Class**: This is an abstract class that provides a blueprint for all specific dataset classes. It defines essential methods like `load_data`, `get_train_data`, `get_test_data`, etc.
  
- **Specific Dataset Classes**: For each supported dataset, there's a corresponding class (e.g., `StrategyQADataset`, `SquadDataset`). These classes inherit from the `Dataset` class and implement dataset-specific loading and processing logic.

- **`prepare_dataset_to_retriever` Method**: This method, available in specific dataset classes, prepares the dataset in a format suitable for the retriever model.

#### Flow:

1. **Initialization**: When a specific dataset class (e.g., `StrategyQADataset`) is instantiated, it loads the dataset from the source and processes it.
  
2. **Data Retrieval**: Methods like `get_train_data` and `get_test_data` provide access to training and testing data, respectively.

3. **Data Preparation for Retriever**: Before using the retriever model, the dataset needs to be prepared in a specific format. The `prepare_dataset_to_retriever` method handles this transformation.

#### Supported Datasets:

- **Strategy QA**: A dataset focused on strategy game-related questions.
  
- **SQuAD**: The Stanford Question Answering Dataset, a popular dataset for machine comprehension tasks.

- **Open TDB**: A trivia database containing a wide range of questions.

- **AQUA RAT**: A dataset containing questions that require arithmetic reasoning.

To use a specific dataset, instantiate its corresponding class and use its methods to access and process the data.

### LLM Model Module
### Retriever Model Module
### Environment Module
### Policy Search Module







### Usage

1. **Training**:
   - Use the `PolicyGradient` or `PPO` class to train the system.
   - Configure the training parameters using the argument parser in `utils/arg_parser.py`.

2. **Retrieving Documents**:
   - Use the retriever models in the `retriever_model` module to fetch relevant documents.
   - Different retriever models like SBERT, BERT, etc., are available.

3. **Generating Answers**:
   - Use the language models in the `llm_model` module to generate answers based on the retrieved documents.

### Dependencies

- PyTorch
- Sentence Transformers
- WandB (for logging and visualization)
- Transformers

### Logging

- The project uses Python's logging module for logging.
- Logs can be directed to both a file and the console.

### Future Work

- Extend the project to support more datasets and models.
- Improve the efficiency and accuracy of the retriever models.
- Experiment with different reinforcement learning algorithms.
