# ML Regression and Discriminant Analysis

This repository contains the implementation of various regression and classification techniques as part of a machine learning assignment. The tasks involve experimenting with Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and regression techniques including Ordinary Least Squares (OLS), Ridge Regression, and Non-linear Regression. 

## Table of Contents
1. [Overview](#overview)
2. [Implemented Methods](#implemented-methods)
3. [Installation](#installation)
4. [Usage](#usage)



---

## Overview
The main focus of this project is to:
1. Implement LDA and QDA for classification and analyze the decision boundaries.
2. Explore regression techniques, calculate Mean Squared Error (MSE), and compare approaches.
3. Optimize Ridge Regression parameters using gradient descent and analyze the effect of regularization.
4. Experiment with Non-linear Regression using polynomial feature mappings.

---

## Implemented Methods
### Gaussian Discriminators
- **Functions**: `ldaLearn`, `qdaLearn`, `ldaTest`, `qdaTest`
- **Objective**: Implement and compare LDA and QDA for classification tasks.
- **Results**: Plots of decision boundaries and accuracy comparison.

### Ordinary Least Squares (OLS) Regression
- **Functions**: `learnOLERegression`, `testOLERegression`
- **Objective**: Minimize squared loss to estimate regression parameters and compare performance with/without intercept.

### Ridge Regression
- **Functions**: `learnRidgeRegression`
- **Objective**: Minimize regularized squared loss, analyze MSE for varying regularization parameter (λ).

### Ridge Regression with Gradient Descent
- **Functions**: `regressionObjVal`, `minimize` (Scipy)
- **Objective**: Implement Ridge Regression using gradient descent and compare results with direct minimization.

### Non-linear Regression
- **Functions**: `mapNonLinear`
- **Objective**: Experiment with polynomial feature mappings and analyze the impact of regularization on model performance.


---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ankit-Mukherjee/ML_Regression_Discriminants.git
   cd ML_Regression_Discriminants
2. Install necessary libraries
    ```bash
   pip install numpy scipy matplotlib

## Usage
- **Prerequisites**:
  - Ensure the following files are present in the working directory:
    - `sample.pickle`
    - `diabetes.pickle`

- **To Run the Script**:
  - Use the command:
    ```bash
    python script.py
    ```

- **Outputs**:
  - Printed accuracy and Mean Squared Error (MSE) values in the console.
  - Generated plots:
    - Decision boundaries.
    - Error trends.

