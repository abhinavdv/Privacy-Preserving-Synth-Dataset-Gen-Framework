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

## Usage
### Data Processing
- Use the scripts in the `docs/` directory for data extraction and processing.

### Model Training
- Fine-tune models using scripts and notebooks in the `framework/` directory.

### Evaluation
- Evaluate model performance using scripts in the `evaluation/` directory.

## Acknowledgments
- Special thanks to all contributors and collaborators.



