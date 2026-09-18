# Adversarial Optimization with Frank-Wolfe Algorithms

This project implements and compares different Frank-Wolfe variants for adversarial attacks on deep neural networks. The goal is to evaluate the effectiveness and efficiency of three Frank-Wolfe algorithms in generating adversarial examples.

## Overview

Adversarial examples are inputs to machine learning models that an attacker has intentionally designed to cause the model to make a mistake. This project focuses on **targeted adversarial attacks** using Frank-Wolfe optimization algorithms, which are particularly well-suited for constrained optimization problems.

## Implemented Algorithms

### 1. Frank-Wolfe Vanilla (`fw_baseline.py`)
- **Description**: Basic Frank-Wolfe algorithm with linear oracle
- **Key Features**: Simple implementation, good baseline performance
- **Use Case**: Standard adversarial attack generation

### 2. Frank-Wolfe with Away Steps (`fw_away_step.py`)
- **Description**: Enhanced Frank-Wolfe with away steps for better convergence
- **Key Features**: Improved convergence rate, better exploration of feasible region
- **Use Case**: Situations requiring faster convergence

### 3. Frank-Wolfe Pairwise (`fw_pairwise.py`)
- **Description**: Pairwise Frank-Wolfe algorithm combining forward and away steps
- **Key Features**: Optimal vertex selection, enhanced performance
- **Use Case**: High-precision adversarial example generation

## Project Structure

```
adversarial_optimization/
├── README.md
├── pyproject.toml
├── poetry.lock
├── .gitignore
├── adversarial_attack.ipynb          # Main experiment notebook
│
├── attack_results/                   # Consolidated results from all experiments
│   ├── fairface_fw_attacks_results.csv  # FairFace attack metrics and data
│   ├── imagenet_fw_attacks_results.csv  # ImageNet attack metrics and data
│   └── mnist_fw_attacks_results.csv     # MNIST attack metrics and data
│
├── plots/                            # Generated visualization plots
│
├── adversarial_optimization/
│   ├── __init__.py
│   ├── image_utils.py                # Image processing and dataset utilities
│   ├── model_utils.py                # Model prediction and target utilities
│   ├── plot_utils.py                 # Visualization and plotting helpers
│   └── algorithms/
│       ├── __init__.py
│       ├── attacks.py                # Main attack coordination function
│       ├── fw_baseline.py            # Vanilla Frank-Wolfe implementation
│       ├── fw_away_step.py           # Frank-Wolfe with Away Steps
│       └── fw_pairwise.py            # Pairwise Frank-Wolfe implementation
└── tests/
    └── __init__.py
```

## Models and Datasets used

### Models
- **ResNet-50**: Pre-trained on ImageNet-1K (1000 classes)
- **MNIST CNN**: SigLIP-based model for digit classification (10 classes)
- **Vision Transformer**: Age classification on FairFace dataset (9 classes)

### Datasets
- **ImageNet-1K**: Large-scale image classification (validation split)
- **MNIST**: Handwritten digit recognition
- **FairFace**: Age classification dataset

## Key Features

### Attack Configuration
- **Epsilon**: Attack strength (L∞ norm constraint)
- **Max Iterations**: Maximum optimization steps
- **Target Labels**: Randomly selected targets different from true labels

### Evaluation Metrics
- **Attack Success Rate**: Percentage of successful attacks
- **Convergence Speed**: Number of iterations to convergence
- **Execution Time**: Time taken per attack
- **Model Accuracy**: Original model performance

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd adversarial_optimization
```

2. **Install dependencies using Poetry:**
```bash
poetry install
```

3. **Activate the virtual environment:**
```bash
poetry shell
```

4. **Launch Jupyter notebook:**
```bash
jupyter notebook adversarial_attack.ipynb
```

## Usage

### Basic Attack Example

```python
from adversarial_optimization.algorithms.fw_baseline import fw_vanilla
from adversarial_optimization.algorithms.fw_away_step import fw_away_steps
from adversarial_optimization.algorithms.fw_pairwise import fw_pairwise

# Attack parameters
epsilon = 0.05
max_iter = 10

# Generate adversarial example
attacked_image, n_iterations = fw_vanilla(
    model=model,
    img_ori=original_image,
    target=target_label,
    epsilon=epsilon,
    max_iter=max_iter
)
```


### Performance Comparison

| Dataset  |   Algorithm   | Success Rate | Avg. Iterations | Avg. Time (s) |
|----------|---------------|--------------|-----------------|---------------|
| ImageNet |  FW Vanilla   |     83.5%    |       12.4      |     0.184     |
| ImageNet | FW Away Steps |     89.7%    |       9.8       |     0.213     |
| ImageNet |  FW Pairwise  |     91.2%    |       8.5       |     0.241     |
|----------|---------------|--------------|-----------------|---------------|
|  MNIST   |  FW Vanilla.  |     77.3%    |       11.6      |     0.076     |
|  MNIST   | FW Away Steps |     82.5%    |       8.9       |     0.095     |
|  MNIST   |  FW Pairwise  |     84.1%    |       8.1       |     0.112     |
|----------|---------------|--------------|-----------------|---------------|
| FairFace |  FW Vanilla.  |     71.8%    |       14.3      |     0.165     |
| FairFace | FW Away Steps |     78.4%    |       11.7      |     0.198     |
| FairFace |  FW Pairwise  |     80.2%    |       10.5      |     0.219     |
|----------|---------------|--------------|-----------------|---------------|

*Note: Results above are for epsilon=0.3. Check the CSV files in `attack_results/` for results with other epsilon values.*

## Configuration

### Attack Parameters
- **Epsilon**: Range from 0.1 to 0.5 (L∞ constraint)
- **Max Iterations**: 25 (15 for testing)
- **Number of Samples**: 200 images per dataset

### Model Settings
- **Preprocessing**: Standard ImageNet normalization
- **Input Size**: 224×224 (ResNet), 28×28 (MNIST)
- **Target Selection**: Random non-true labels

## File Outputs

The experiment generates several output files in the `attack_results/` directory:
- `fairface_fw_attacks_results.csv`: Results from FairFace attacks with ViT model
- `mnist_fw_attacks_results.csv`: Results from MNIST attacks with SigLIP model
- Additional result files are generated with different epsilon values


## References

- Frank-Wolfe Algorithm: Frank, M., & Wolfe, P. (1956)
- Away Steps: Lacoste-Julien, S., & Jaggi, M. (2015)
- Pairwise Frank-Wolfe: Lacoste-Julien, S., & Jaggi, M. (2015)

## Contact

- **Authors**: Arturo Bevilacqua, Mattia Piazza
- **Course**: Optimization for Data Science

---

*This project implements Frank-Wolfe variants for adversarial attacks as part of university coursework in optimization methods.*
