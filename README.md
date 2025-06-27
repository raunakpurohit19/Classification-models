# Machine Learning for Medical Diagnosis

This repository contains a collection of Jupyter Notebooks demonstrating various machine learning and deep learning techniques for predicting medical conditions. The projects serve as practical examples of data preprocessing, model building, and evaluation on real-world medical datasets.

## 📋 Table of Contents
*   [Projects Overview](#-projects-overview)
*   [Datasets Used](#-datasets-used)
    *   [Pima Indians Diabetes Dataset](#-pima-indians-diabetes-dataset)
    *   [Heart Failure Prediction Dataset](#-heart-failure-prediction-dataset)
*   [Technologies Used](#-technologies-used)
*   [Setup and Installation](#-setup-and-installation)
*   [How to Run](#-how-to-run)
*   [Project Details](#-project-details)
    *   [1. Diabetes Prediction Models](#1-diabetes-prediction-models)
    *   [2. Heart Disease Prediction](#2-heart-disease-prediction)
*   [Key Findings & Comparison](#-key-findings--comparison)
*   [Future Improvements](#-future-improvements)

## 🚀 Projects Overview

1.  **Diabetes Prediction (`Diabetes Prediction Models.ipynb`)**: This notebook compares traditional machine learning models (Logistic Regression, Random Forest) with a simple Neural Network to predict the onset of diabetes.
2.  **Heart Disease Prediction (`heart.ipynb`)**: This notebook uses several classification models, including K-Nearest Neighbors and Random Forest, to predict the presence of heart disease based on clinical features. It also includes data preprocessing steps like label encoding.

## 📊 Datasets Used

### 🩺 Pima Indians Diabetes Dataset
*   **Source**: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
*   **File**: `diabetes.csv`
*   **Description**: This dataset contains diagnostic measurements for female patients of Pima Indian heritage. The goal is to predict the `Outcome` (1 for diabetic, 0 for non-diabetic).
*   **Features**: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.

### ❤️ Heart Failure Prediction Dataset
*   **Source**: [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
*   **File**: `heart.csv`
*   **Description**: This dataset combines five other heart disease datasets and provides a clean set of clinical features to predict the `HeartDisease` target (1 for presence, 0 for absence).
*   **Features**: `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`.

## 🛠️ Technologies Used
*   Python 3
*   Pandas & NumPy (for data manipulation)
*   Scikit-learn (for ML models and preprocessing)
*   TensorFlow & Keras (for the neural network)
*   Matplotlib (for plotting)
*   Jupyter Notebook / Google Colab

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/medical-diagnosis-ml.git
    cd medical-diagnosis-ml
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib
    ```

4.  **Download the datasets:**
    *   Download `diabetes.csv` from the link above and place it in the project's root directory.
    *   Download `heart.csv` from the link above and place it in the project's root directory.

## ▶️ How to Run
1.  Ensure all dependencies are installed and datasets are in the root directory.
2.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open either `Diabetes Prediction Models.ipynb` or `heart.ipynb`.
4.  **Important**: Before running the cells, make sure the file paths in the notebooks are correct.

    *   In `Diabetes Prediction Models.ipynb`, change:
        ```python
        # From (if using Colab)
        df = pd.read_csv('/content/diabetes (1).csv')
        # To (for local execution)
        df = pd.read_csv('diabetes.csv')
        ```

    *   In `heart.ipynb`, change:
        ```python
        # From (an absolute path)
        data = pd.read_csv(r"D:\\...\\heart.csv")
        # To (for local execution)
        data = pd.read_csv('heart.csv')
        ```
5.  Run the cells in the notebook from top to bottom.

## 📝 Project Details

### 1. Diabetes Prediction Models
*   **File**: `Diabetes Prediction Models.ipynb`
*   **Objective**: Predict the onset of diabetes.
*   **Models Implemented**:
    1.  Logistic Regression
    2.  Random Forest Classifier
    3.  A simple feed-forward Neural Network (TensorFlow/Keras)
*   **Workflow**:
    1.  **Data Loading**: Loads `diabetes.csv`.
    2.  **Preprocessing**: Splits data into training and testing sets. **Note**: The features are not scaled, which impacts the neural network's performance.
    3.  **Model Training**: Each of the three models is trained on the same dataset.
    4.  **Evaluation**: The accuracy of each model is calculated and printed.
*   **Results**:
    *   Logistic Regression Accuracy: **~74.0%**
    *   Random Forest Accuracy: **~74.7%**
    *   Neural Network Accuracy: **~66.9%** (This lower score highlights the importance of feature scaling for NNs).

### 2. Heart Disease Prediction
*   **File**: `heart.ipynb`
*   **Objective**: Predict the presence of heart disease.
*   **Models Implemented**:
    1.  K-Nearest Neighbors (with a loop to find the optimal `k`)
    2.  Random Forest Classifier
    3.  Logistic Regression
*   **Workflow**:
    1.  **Data Loading**: Loads `heart.csv`.
    2.  **Preprocessing**:
        *   Categorical features (`Sex`, `ChestPainType`, etc.) are converted to numerical format using `LabelEncoder`.
        *   Data is split into training and testing sets using `stratify=y` to maintain the same class distribution in both sets.
    3.  **Visualization**: A `scatter_matrix` is generated to visualize relationships between all features.
    4.  **Model Training & Evaluation**: The models are trained and their accuracy is printed.
*   **Results**:
    *   K-Nearest Neighbors (k=11): **~73.9%**
    *   Random Forest Accuracy: **~87.3%**
    *   Logistic Regression Accuracy: **~86.6%**

## 🔬 Key Findings & Comparison
*   The **Random Forest** model performed strongly on both datasets, demonstrating its robustness.
*   The **Heart Disease** notebook shows better preprocessing practices (`LabelEncoder`, `stratify`), leading to higher model accuracies.
*   The **Diabetes** notebook provides a clear case study on why **feature scaling** is crucial for neural networks. The unscaled data likely prevented the NN from converging effectively, resulting in lower accuracy compared to simpler models.

## 🚀 Future Improvements
This repository can be extended with the following improvements:

*   **Feature Scaling**: Apply `StandardScaler` to the data in *both* notebooks, especially before training the neural network and logistic regression models.
*   **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for all models to maximize their performance.
*   **Deeper EDA**: Add more detailed Exploratory Data Analysis, including correlation heatmaps and distribution plots, to gain more insights from the data.
*   **Advanced Models**: Implement more sophisticated models like Gradient Boosting (XGBoost, LightGBM) or more complex neural network architectures.
*   **Robust Evaluation**: Use k-fold cross-validation for more reliable model evaluation and include metrics like **Precision, Recall, F1-Score**, and the **Confusion Matrix**.
