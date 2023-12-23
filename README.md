### Project Overview

This project is a reinforcement learning-based approach to optimize the performance of a question-answering system. The system uses a retriever model to fetch relevant documents and a language model to generate answers based on the retrieved documents. The policy gradient method is employed to train the system, with the option to use the Proximal Policy Optimization (PPO) algorithm.

### Table Of Contents

1. [**Dependencies**](#dependencies)
2. [**Getting Started**](#getting-started)
3. [**Quick Guide to CLI Arguments**](#quick-guide-to-cli-arguments)
4. [**Modules**](#modules)

### Dependencies

- PyTorch
- Sentence Transformers
- WandB (for logging and visualization)
- Transformers

### Getting Started

To get started with this question-answering system, follow these steps:

1. **Install Dependencies**: Ensure you have Python 3 installed. Then, install the required libraries using pip:
   ```bash
   pip install torch sentence-transformers wandb transformers
   ```
2. **Clone the Repository**: Clone the project repository to your local machine.
   ```bash
   git clone <repository_url>
   ```
3. **Set Environment Variables**: Set the necessary environment variables. Replace `<your_key>` with your actual keys.
   ```bash
   export OPENAI_API_KEY=<your_openai_api_key>
   export HF_TOKEN=<your_huggingface_token>
   export WANDB_API_KEY=<your_wandb_api_key>
   ```

4. **Run the Code**: Navigate to the project directory and run the main script.
   ```bash
   cd NLP_project_sftp
   python3 main.py --<arguments>
   ```

### Example Run

Here's an example of how to run the system with a specific set of arguments:
   ```bash
   python3 main.py --dataset aqua-rat --llm_model gpt3.5 --num_batches 500 --retriever_model sbert --retriever_top_k 3 --normalize_encoding_method l2 --algorithm pg --baseline --llm_max_prompt_tokenized_len 700 --llm_max_output_tokenized_len 400 --first_layer_size 1024 --learning_rate 0.01 --n_layers 5 --run_name aqua_rat_exploration_0.995_top_3 --num_episodes_per_batch 20 --exploration_decay_factor 0.995 --policy_exploration_logic epsilon_greedy
   ```

#### Parameters Explained:

- **`--dataset aqua-rat`**: Utilizes the AQUA RAT dataset.
- **`--llm_model gpt3.5`**: Employs GPT-3.5 as the language model.
- **`--retriever_model sbert`**: Uses Sentence-BERT for document retrieval.
- **`--algorithm pg`**: Chooses the Policy Gradient algorithm.
- **`--baseline`**: Includes a baseline network for policy learning.

### Quick Guide to CLI Arguments

#### General Configuration
- **`--run_name`**: Optional name for the run, useful for logging and tracking.
- **`--seed`**: Random seed for reproducibility.

#### Dataset Configuration
- **`--dataset`**: The dataset to use (e.g., 'strategy-qa', 'squad').

#### Retriever Configuration
- **`--retriever_model`**: The retriever model (e.g., 'sbert', 'bert-no-op-retriever').
- **`--normalize_encoding_method`**: Method to normalize encodings ('l2', 'instance', or none).
- **`--retriever_top_k`**: Top K documents to retrieve (for retrievers that support it).

#### Language Learning Model (LLM) Configuration
- **`--llm_model`**: The language learning model (e.g., 'gpt2', 'gpt3.5').
- **`--llm_max_prompt_tokenized_len`**: Max token length for the LLM prompt.
- **`--llm_max_output_tokenized_len`**: Max token length for the LLM output.
- **`--llm_temperature`**: Temperature for LLM's softmax.

#### Policy and Training Configuration
- **`--algorithm`**: The training algorithm ('pg' for Policy Gradient, 'ppo' for Proximal Policy Optimization).
- **`--gamma`**: Discount factor for the policy gradient algorithm.
- **`--baseline`**: Whether to use a baseline in the policy gradient.
- **`--policy_instance_norm`**: Whether to use instance normalization in the policy network.
- **`--n_layers`**: Number of layers in the policy network.
- **`--first_layer_size`**: Size of the first layer in the policy network.
- **`--learning_rate`**: Learning rate for the optimizer.
- **`--normalize_advantage`**: Whether to normalize the advantage function.
- **`--num_batches`**: Number of batches to train on.
- **`--num_episodes_per_batch`**: Number of episodes per batch.
- **`--test_every`**: Frequency of testing the model.

#### Proximal Policy Optimization (PPO) Specific Configuration
- **`--eps_clip`**: Epsilon clip parameter for PPO.
- **`--update_freq`**: Update frequency for PPO.

#### Policy Exploration Configuration
- **`--policy_exploration_logic`**: Exploration logic for the policy ('epsilon_greedy', 'linear_temperature_decay', 'exponential_temperature_decay').
- **`--initial_temperature`**: Initial temperature for policy exploration.
- **`--end_temperature`**: End temperature for policy exploration.
- **`--exploration_decay_factor`**: Decay factor for policy exploration.

### Modules

1. [**Dataset**](#dataset-module)
2. [**LLM Model**](#llm-model-module)
3. [**Retriever Model**](#retriever-model-module)
4. [**Environment**](#environment-module)
5. [**Policy Search**](#policy-search-module)

### Dataset Module

The Dataset module is responsible for managing and providing access to various datasets. 

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

The LLM (Language Learning Model) module is integral to the question-answering system, responsible for generating answers based on the information retrieved by the retriever model. 

#### Functionality:

- **`LLMModel` Class**: An abstract base class that outlines the structure for all LLM models. It initializes common parameters like maximum tokenized lengths and temperature settings for answer generation.

- **Specific LLM Classes**: Each class represents a specific LLM, such as GPT-2, GPT-3.5, FlanT5, or Llama models. These classes inherit from `LLMModel` and implement model-specific functionalities.
  - **`generate_answer` Method**: An abstract method overridden in each subclass to generate answers based on a given prompt.

#### Flow:

1. **Initialization**: When an LLM class (e.g., `GPT2LLM`) is instantiated, it loads the specific model and tokenizer, setting up necessary configurations.

2. **Answer Generation**: The `generate_answer` method in each LLM class takes a prompt (usually a question and context) and generates an answer using the underlying language model.

#### Supported LLM Models:

- GPT-2
- GPT-3.5 Turbo
- FlanT5 Small
- FlanT5 Base
- FlanT5 Large
- FlanT5 XL
- Llama-2-7b

### Retriever Model Module

The Retriever Model module is responsible for fetching relevant information from the dataset based on the query provided by the user or the system.

#### Functionality:

- **`RetrieverModel` Class**: This abstract base class defines the common structure and functionalities for all retriever models. It includes methods for encoding inputs and retrieving relevant data.

- **`SBertRetriever` Class**: A specific implementation of the `RetrieverModel`, using the Sentence-BERT model to retrieve semantically relevant documents or passages. It is designed to work with datasets that can be encoded into a vector space for similarity comparison.

#### Flow:

1. **Initialization**: The `SBertRetriever` class, when instantiated, loads the Sentence-BERT model and prepares the dataset for retrieval by encoding the documents.

2. **Encoding**: The `encode` method converts the input query into a vector representation using the Sentence-BERT model.

3. **Retrieval**: The `retrieve` method takes the encoded query and finds the top K most relevant documents from the dataset, based on semantic similarity.

#### Supported Retriever Model:

- **SBertRetriever**: Utilizes the Sentence-BERT model for semantic search and retrieval, offering efficient and effective retrieval of information from large datasets.

### Environment Module

The Environment module in the question-answering system serves as the interface between the agent (policy model) and the question-answering components (dataset, retriever, and LLM).

#### Functionality:

- **`Environment` Class**: Manages the interactions and flow of data between the agent and the question-answering components.
  - **Initialization**: Sets up the dataset, retriever, LLM, and defines the action and observation spaces.
  - **`step` Method**: Processes the agent's action, updates the environment's state, and returns the next observation, reward, and a flag indicating if the episode has ended.
  - **`reset` Method**: Resets the environment to a new initial state for each episode.
  - **`evaluate_prompt` Method**: Generates an answer using the LLM and evaluates it against the ground truth to calculate the reward.

#### Flow:

1. **Initialization**: The environment is initialized with the dataset, retriever, LLM, and other parameters like the action space (based on the retriever's top K documents) and observation space.

2. **Episode Start**: At the beginning of each episode, `reset` is called to fetch a new question and set the initial state.

3. **Agent Interaction**:
   - The agent receives an observation from the environment.
   - Based on this observation, the agent decides on an action (e.g., selecting a document).
   - The agent's action is passed to the `step` method.

4. **Environment Response**:
   - The environment processes the action, updates the context prompt, and checks if the LLM's prompt limit is exceeded.
   - If the episode is done (either by reaching the prompt limit or by a termination action), the environment evaluates the prompt and calculates the reward.
   - The environment returns the next observation, the reward, and a flag indicating if the episode has ended.

5. **Episode End**: The episode ends when the maximum number of steps is reached or a specific terminal condition is met. The agent then starts a new episode by calling `reset`.

The Environment module simulates the question-answering scenario, allowing the agent to learn and adapt its policy based on the feedback received from the interactions.

### Policy Search Module

The Policy Search module is a critical component of the question-answering system, responsible for determining the best actions to take based on the current state of the environment. It encompasses the implementation of policy gradient algorithms, including Policy Gradient (PG) and Proximal Policy Optimization (PPO).

#### Functionality:

- **`BasePolicy` Class**: An abstract class that defines the structure for policy models. It includes methods for determining action distributions and selecting actions based on observations.

- **`CategoricalPolicy` Class**: A specific implementation of `BasePolicy` that uses categorical distributions for action selection. It is particularly suited for environments with discrete action spaces. This class implements a multi-layer perceptron (MLP) model, where the architecture can be configured with varying numbers of layers and units per layer.

- **`PolicyGradient` Class**: Implements the policy gradient algorithm, a reinforcement learning approach that optimizes the policy directly.

- **`PPO` Class**: Extends `PolicyGradient` to implement the Proximal Policy Optimization algorithm, which improves training stability and efficiency by limiting the extent of policy updates.

- **`BaselineNetwork` Class**: An optional component used in policy gradient methods to estimate the value function, helping to reduce the variance of policy updates. The baseline network, also an MLP, predicts the expected return from each state, allowing the calculation of advantages (the difference between actual returns and baseline predictions) to guide more effective policy updates.

#### Flow:

1. **Policy Initialization**: The policy (either `CategoricalPolicy` for PG or a variant for PPO) is initialized with the network architecture and training configurations. The MLP architecture of the policy is flexible, allowing customization in terms of layers and neuron counts.

2. **Training Episodes**: The policy interacts with the environment, collecting data on states, actions, and rewards.

3. **Policy Update**:
   - For PG: The policy is updated using the collected data to maximize the expected return.
   - For PPO: The policy update is constrained to ensure small changes, improving training stability.

4. **Baseline Network**: If used, the baseline network provides an estimate of the value function, which is crucial for calculating advantages and guiding the policy update process.

5. **Evaluation**: The policy is periodically evaluated on the test set to monitor its performance.

The Policy Search module is essential for learning an effective strategy to navigate the question-answering environment, adapting the policy based on the feedback received from interactions with the environment. The use of MLPs in both the policy and baseline network allows for a flexible and adaptable approach to learning the optimal policy.

