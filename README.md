# Weight Initialization in Neural Networks

## 📋 Project Overview
This project explores the critical role of weight initialization in training deep neural networks. Through hands-on experimentation with a 3-layer neural network, we demonstrate how different initialization techniques dramatically affect model performance, convergence speed, and final accuracy.

## 🎯 Learning Objectives
- Understand why proper weight initialization matters
- Compare zero, random, and He initialization methods
- Observe the effects of symmetry breaking in neural networks
- Learn to implement different initialization strategies
- Analyze the impact of initialization on gradient flow and convergence

## 🧠 Initialization Methods Explored

### 1. Zero Initialization
- **Implementation**: All weights and biases set to zero
- **Problem**: Fails to break symmetry - all neurons learn identical features
- **Result**: Network performs no better than logistic regression (~50% accuracy)

### 2. Random Initialization (Large Values)
- **Implementation**: Weights scaled by factor of 10
- **Problem**: Extremely large weights cause vanishing/exploding gradients
- **Result**: Unstable training, initial cost = inf, but achieves ~83% accuracy

### 3. He Initialization (Recommended)
- **Implementation**: Weights scaled by √(2/n_l_prev)
- **Advantage**: Optimized for ReLU activations, maintains proper gradient flow
- **Result**: Fast convergence, excellent performance (~99% accuracy)

## 📊 Key Findings

| Initialization | Train Accuracy | Convergence | Issue/Solution |
|----------------|---------------|-------------|----------------|
| Zeros | 50% | No convergence | Symmetry breaking fails |
| Random (×10) | 83% | Slow, unstable | Gradients explode/saturate |
| He | 99% | Fast, stable | Optimal for ReLU |

## 🛠️ Technical Implementation

### Dependencies
Install dependencies
requirements.txt
numpy>=1.21.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
jupyter>=1.0.0
ipython>=7.0.0

bash
pip install -r requirements.txt
Launch Jupyter Notebook

bash
jupyter notebook Weight_Initialization.ipynb 
 Results Visualization
The notebook generates three key visualizations:

Cost curves showing convergence speed for each method

Decision boundaries illustrating classification performance

Accuracy comparisons between initialization techniques

🔬 Key Insights
Symmetry Breaking: Zero initialization prevents neurons from learning different features

Scale Sensitivity: Weight magnitude critically affects gradient flow

Activation Matching: Initialization should match activation function (He → ReLU)

Convergence Speed: Proper initialization can dramatically reduce training time


