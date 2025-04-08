#  Customer Churn Prediction with MLlib

This project uses Apache Spark MLlib to predict customer churn based on structured customer data. You will preprocess data, train classification models, perform feature selection, and tune hyperparameters using cross-validation.

---

Build and compare machine learning models using PySpark to predict whether a customer will churn based on their service usage and subscription features.

---

##  Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

---

##  Tasks

### Task 1: Data Preprocessing and Feature Engineering

**Objective:**  
Clean the dataset and prepare features for ML algorithms.

**Steps:**
1. Fill missing values in `TotalCharges` with 0.
2. Encode categorical features using `StringIndexer` and `OneHotEncoder`.
3. Assemble numeric and encoded features into a single feature vector with `VectorAssembler`.

**Explanation of code:**
1. The dataset is loaded and missing values in the TotalCharges column are filled with zero to handle data quality issues.
2. Categorical columns are encoded using StringIndexer followed by OneHotEncoder to convert them into numerical format.
3. Numeric and encoded categorical columns are then assembled into a single feature vector using VectorAssembler.
This feature vector will be used as input for machine learning models.

**Code Output:**

```
+--------------------+-----------+
|features            |ChurnIndex |
+--------------------+-----------+
|(8,[1,2,3,7],[2.0,103.27,215.86,1.0])|0.0        |
|[1.0,18.0,87.06,1440.26,0.0,1.0,1.0,0.0]|0.0        |
|(8,[1,2,3,6],[62.0,40.41,2963.95,1.0])|0.0        |
|(8,[1,2,3,6],[39.0,53.78,1719.86,1.0])|0.0        |
|(8,[1,2,3,6],[10.0,72.63,629.16,1.0])|1.0        |
+--------------------+-----------+
```
---

### Task 2: Train and Evaluate Logistic Regression Model

**Objective:**  
Train a logistic regression model and evaluate it using AUC (Area Under ROC Curve).

**Steps:**
1. Split dataset into training and test sets (80/20).
2. Train a logistic regression model.
3. Use `BinaryClassificationEvaluator` to evaluate.

**Explanation of code:**
1. The dataset is split into training (80%) and test (20%) sets using randomSplit.
2. A LogisticRegression model is trained using the features vector and ChurnIndex label.
3. The trained model is used to make predictions on the test data.
4. Performance is evaluated using BinaryClassificationEvaluator based on AUC (Area Under the ROC Curve).

**Code Output Example:**
```
Logistic Regression Model Accuracy (AUC): 0.72
```

---

###  Task 3: Feature Selection using Chi-Square Test

**Objective:**  
Select the top 5 most important features using Chi-Square feature selection.

**Steps:**
1. Use `ChiSqSelector` to rank and select top 5 features.
2. Print the selected feature vectors.

**Explanation of code:**
1. ChiSqSelector is used to perform statistical tests between input features and the target label.
2. It ranks the features based on their importance in predicting churn.
3. Only the top 5 most relevant features are selected to reduce dimensionality.
4. The transformed dataset with selected features is displayed for verification.

**Code Output Example:**
```
+--------------------+-----------+
|selectedFeatures    |ChurnIndex |
+--------------------+-----------+
|(5,[1,4],[2.0,1.0]) |0.0        |
|[1.0,18.0,1.0,1.0,0.0]|0.0        |
|(5,[1,3],[62.0,1.0])|0.0        |
|(5,[1,3],[39.0,1.0])|0.0        |
|(5,[1,3],[10.0,1.0])|1.0        |
+--------------------+-----------+

```

---

### Task 4: Hyperparameter Tuning and Model Comparison

**Objective:**  
Use CrossValidator to tune models and compare their AUC performance.

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosted Trees (GBT)

**Steps:**
1. Define models and parameter grids.
2. Use `CrossValidator` for 5-fold cross-validation.
3. Evaluate and print best model results.

**Explanation of code:**
1. Four classification models—Logistic Regression, Decision Tree, Random Forest, and GBT—are defined along with their hyperparameter grids.
2. Each model is tuned using CrossValidator with 5-fold cross-validation to find the best-performing parameters.
3. The best model from each tuning process is used to predict on the test set, and AUC is calculated.
4. The model name, best AUC score, and optimal parameters are written to a result file for comparison.

**Code Output Example:**
```
Tuning LogisticRegression...
LogisticRegression Best Model Accuracy (AUC): 0.71
Best Params for LogisticRegression: maxIter=20, regParam=0.01

Tuning DecisionTree...
DecisionTree Best Model Accuracy (AUC): 0.58
Best Params for DecisionTree: maxDepth=5

Tuning RandomForest...
RandomForest Best Model Accuracy (AUC): 0.76
Best Params for RandomForest: maxDepth=10, numTrees=50

Tuning GBT...
GBT Best Model Accuracy (AUC): 0.78
Best Params for GBT: maxDepth=5, maxIter=20

```
---

##  Execution Instructions

### 1. Prerequisites

- Apache Spark installed
- Python environment with `pyspark` installed
- `customer_churn.csv` placed in the project directory

### 2. Run the Project
- Dataset generation
```bash
python dataset-generator.py
```
```bash
spark-submit customer-churn-analysis.py
```