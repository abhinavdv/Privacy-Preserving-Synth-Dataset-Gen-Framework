# Synthetic Data Generation Framework for Privacy-Preserving Production Data Enhancing LLM Performance

## Project Overview


In today’s data-driven landscape, organizations increasingly rely on large language models (LLMs) for a wide range of tasks, including natural language processing, customer service automation, and data analysis. However, the use of real-world data for training these models presents significant privacy concerns, especially when sensitive information is involved. Traditional data anonymization techniques often lead to substantial information loss, which compromises the utility of the data. To address this challenge, we propose developing a synthetic data generation framework that ensures privacy preservation while maintaining the statistical properties of original datasets. This framework will enable organizations to train LLMs effectively without exposing sensitive information, enhancing both data security and model performance.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Data License

Dataset used is a large-scale Amazon Reviews dataset, collected in 2023 by McAuley Lab. **Copyright © 2024, McAuley Lab. All rights reserved.** Usage of this dataset must comply with the terms set by the original authors. 

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

### Prerequisites
- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
## Directory Structure
```
.gitignore
LICENSE
README.md
requirements.txt
docs/
  img/
    architecture.jpg
framework/
  [1]_procesed_input_data/
    train.jsonl
    test.jsonl
  [2]_generated_synthetic_data/
    generated_sequences_with_dp.jsonl
    generated_sequences_no_dp.jsonl
  [0]_process_amazon_data.ipynb 
  [1]_finetuning_with_tracking.ipynb
  [2]_inferenceing_finetuned_llm.ipynb
evaluation/
  privacy/
    [1]_canary_sequence.ipynb
    [2]_finetuning_with_tracking_canary.ipynb
    canary_injected_data/
  fidelity/
    [1]_fidelity_evaluation.ipynb
  utility/
    [1]_downstream_llm_finetuning.ipynb
    [2]_feature_importance_consistency_evaluation.ipynb
```

## Project Workflow
The project workflow consists of the following steps:

### Data Acquistion
Data was sourced from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), a large-scale dataset containing product reviews and metadata. This project uses specifically the "Cell Phones and Accessories" subset. The dataset includes:
- Reviews Data: Cell_Phones_and_Accessories.jsonl (customer reviews)
- Products Metadata: meta_Cell_Phones_and_Accessories.jsonl (product descriptions)

### Data Processing
1. [[0]_process_amazon_data.ipynb](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/framework/%5B0%5D_process_amazon_data.ipynb) - Loads raw Amazon reviews and product descriptions (.jsonl format). Cleans, filters, and processes the text data into structured prompt-response pairs. Saves the processed data into [train.jsonl](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/framework/%5B1%5D_procesed_input_data/train.jsonl) and [test.jsonl](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/framework/%5B1%5D_procesed_input_data/test.jsonl).

### Synthetic Data Generation Framework (Model Fine-Tuning & Inference)
2. [[1]_finetuning_with_tracking.ipynb](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/framework/%5B1%5D_finetuning_with_tracking.ipynb) - Loads the processed dataset (train.jsonl). Fine-tunes Llama-3.1-8B using LoRA and Opacus (Differential Privacy). Tracks training with wandb and saves fine-tuned checkpoints.
3. [[2]_inferenceing_finetuned_llm.ipynb](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/framework/%5B2%5D_inferenceing_finetuned_llm.ipynb) - Loads the fine-tuned LLM and performs inference on test.jsonl.Generates synthetic responses both with and without DP. Saves results into [generated_sequences_with_dp.jsonl](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/framework/%5B2%5D_generated_synthetic_data/generated_sequences_with_dp.jsonl) and [generated_sequences_no_dp.jsonl](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/framework/%5B2%5D_generated_synthetic_data/generated_sequences_no_dp.jsonl).

### Privacy Evaluation of Synthetic Data
4. [[1]_canary_sequence.ipynb](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/evaluation/privacy/%5B1%5D_canary_sequence.ipynb) - Injects canary phrases into the training dataset to test privacy leakage. Uses FuzzyWuzzy and Faker to generate realistic canary sequences.
Saves modified datasets (amazon_train_canary_1.jsonl, etc.).
5. [[2]_finetuning_with_tracking_canary.ipynb](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/evaluation/privacy/%5B2%5D_finetuning_with_tracking_canary.ipynb) - Fine-tunes the model with canary-injected data and tests for leakage. Runs LoRA fine-tuning with DP to check if canary phrases appear in outputs.Compares privacy leakage rates in synthetic data.

### Fidelity Evaluation of Synthetic Data
6. [[1]_fidelity_evaluation.ipynb](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/evaluation/fidelity/%5B1%5D_fidelity_evaluation.ipynb) - Evaluates how closely synthetic data resembles real data. Compares feature distributions, token structures, and statistical patterns. Assesses differences between real, non-DP synthetic, and DP synthetic data.

### Utility Evaluation of Synthetic Data
7. [[1]_downstream_llm_finetuning.ipynb](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/evaluation/utility/%5B1%5D_downstream_llm_finetuning.ipynb) - Fine-tunes an LLM on synthetic data to assess its real-world usability. Compares performance between models trained on real vs. synthetic datasets.Uses unsloth for efficient fine-tuning.

8. [[2]_feature_importance_consistency_evaluation.ipynb](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/evaluation/utility/%5B2%5D_feature_importance_consistency_evaluation.ipynb) - Evaluates whether important features in real data remain important in synthetic data.Uses scikit-learn to check feature importance rankings in different datasets. Ensures key attributes are retained even after DP transformations.

    In addition to the above evaluation, human evaluator assessed the quality of synthetic data by comparing it with real data on randomly sampled 500 records.

### Final Output
Privacy-Preserving, Highly Realistic Synthetic Data for LLM Training, with Evaluation Metrics for fidelity, privacy leakage, and utility assessment. 
- [generated_sequences_with_dp.jsonl](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/framework/%5B2%5D_generated_synthetic_data/generated_sequences_with_dp.jsonl) .
- [privacy_evaluation_result.jpg](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/docs/img/privacy_evaluation_result.jpeg)
- [fidelity_evaluation_result.jpg](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/docs/img/fidelity_evaluation_result.jpeg)
- [utility_evaluation_result_1.jpg](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/docs/img/utility_evaluation_result_1.jpeg)
- [utility_evaluation_result_2.jpg](https://github.com/abhinavdv/Privacy-Preserving-Synth-Dataset-Gen-Framework/blob/main/docs/img/utility_evaluation_result_2.jpeg)



