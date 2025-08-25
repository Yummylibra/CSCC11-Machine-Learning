# Classification – MNIST Handwritten Digit Recognition

This branch explores **supervised learning for image classification**, using the **MNIST handwritten digit dataset** as a benchmark. The focus is on understanding how **different models generalize** under clean data, noisy inputs, and noisy labels.  

---

## Theorey

### 1. Classification Problem
Classification seeks to learn a function \( f: X \to Y \) mapping inputs \(x\) (e.g., pixel intensities of an image) to discrete labels \(y \in \{1, \dots, K\}\).  
- In binary classification, the decision boundary is defined by the set of points where the classifier is equally likely to assign two classes.  
- For multi-class classification (e.g., MNIST with 10 digits), the task is to partition the input space into regions corresponding to each digit:contentReference[oaicite:0]{index=0}.  

---

### 2. Models Explored

#### **K-Nearest Neighbors (KNN)**
- Decision for a new input is based on majority voting among the \(k\) closest training samples:contentReference[oaicite:1]{index=1}.  
- Larger \(k\) values smooth the decision boundary, reducing variance but potentially increasing bias.  
- Sensitive to feature scaling and noise, since distances can be distorted by irrelevant or corrupted inputs.  

#### **Random Forest (RF)**
- An ensemble of decision trees, each trained on bootstrapped subsets of data and features:contentReference[oaicite:2]{index=2}.  
- Reduces overfitting by averaging across multiple trees.  
- Hyperparameter choice (e.g., `min_samples_leaf`) controls trade-off between complexity and generalization.  

---

### 3. Learning and Optimization
- Models are fit by minimizing empirical error, e.g., squared loss or classification error.  
- Gradient-based optimization is less relevant here since KNN is non-parametric and RF relies on recursive splits, but the conceptual framework of minimizing a loss function still applies:contentReference[oaicite:3]{index=3}.  

---

### 4. Cross-Validation for Model Selection
- Hyperparameters such as \(k\) in KNN or tree depth in RF cannot be chosen arbitrarily.  
- **K-fold cross-validation** provides an unbiased way to estimate generalization error:contentReference[oaicite:4]{index=4}.  
- The experiments tuned parameters across predefined ranges (e.g., \(k=1,6,11,\dots,96\) for KNN).  

---

## Scenarios

### **Scenario 1 – Clean Data**
- Training and testing on clean MNIST digits.  
- **Observation**:  
  - KNN: Best at large \(k\) (≈96), accuracy ≈ 0.80.  
  - RF: Best at `min_samples_leaf = 5`, accuracy ≈ 0.96.  
- **Lesson**: With high-quality data, flexible models like RF capture structure better, while KNN benefits from smoothing effects.  

---

### **Scenario 2 – Noisy Inputs**
- Test data corrupted by pixel-level noise.  
- **Observation**:  
  - KNN collapses to ≈ 0.21 accuracy.  
  - RF drops to ≈ 0.10 accuracy.  
- **Lesson**: Input noise violates the assumption that nearby points in feature space share labels. Both models fail, highlighting need for preprocessing (e.g., denoising, feature extraction).  

---

### **Scenario 3 – Noisy Labels**
- 20,000 training labels replaced randomly.  
- **Observation**:  
  - KNN: Best at moderate \(k\) (≈61), but accuracy ≈ 0.21.  
  - RF: Best with stronger regularization (`min_samples_leaf=25`), but still ≈ 0.10 accuracy.  
- **Lesson**: Label corruption is devastating. Even complex models cannot recover from incorrect supervision, showing the critical role of data quality.  

---

## Conclusion
- **Bias–variance tradeoff**: Larger \(k\) smooths KNN boundaries; smaller leaves in RF increase variance.  
- **Generalization vs. overfitting**: Clean data enables strong performance; corrupted data exposes weaknesses.  
- **Robustness**: Input noise mainly harms distance-based methods; label noise undermines all supervised approaches.  
- **Practical Implication**: Good ML practice requires not only algorithmic choice but also careful **data preprocessing, denoising, and validation**.  



---

## 📝 Summary
This assignment demonstrates that supervised learning can achieve high accuracy under ideal conditions but is **fragile to noise**. The experiments bridge **theory (classification, estimation, generalization)** with **practice (MNIST benchmark, KNN, RF)**, reinforcing the central role of data quality and robust evaluation in machine learning.  
