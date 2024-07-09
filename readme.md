## Santander Customer Transaction Prediction

### Overview
This project focuses on predicting whether Santander customers will make a transaction, based on anonymized data. The challenge is hosted on Kaggle, and the main objective is to predict binary outcomes using machine learning techniques.

### Tools and Libraries Used
- **Python**, **Pandas**, **NumPy**, **Scikit-Learn** for data manipulation and preliminary modeling.
- **LightGBM**, **XGBoost** for machine learning models.
- **Optuna** for optimizing model configurations.
- **Matplotlib** for data visualization.

### Dataset
The dataset is available directly from Kaggle. You will need to set up a Kaggle account and agree to the competition rules to access the data.

### Downloading the Dataset
1. Go to the [Santander Customer Transaction Prediction competition page on Kaggle](https://www.kaggle.com/c/santander-customer-transaction-prediction).
2. Join the competition by accepting the competition rules.
3. Use the Kaggle API to download the dataset:

```bash
# Ensure you have kaggle installed
pip install kaggle

# Make sure your kaggle.json file is in the correct location (usually under ~/.kaggle/)
kaggle competitions download -c santander-customer-transaction-prediction

# Unzip the downloaded files
unzip santander-customer-transaction-prediction.zip
```

The dataset consists of 200,000 observations, each described by 202 anonymized features. Key preprocessing steps included:

- **Data Cleaning**: Confirmed no missing values present.
- **Feature Removal**: Dropped the `ID_code` as it was an identifier and not predictive.

### Preprocessing
- **Standardization**: Features were scaled using `StandardScaler`.
- **Principal Component Analysis (PCA)**: Reduced dimensions to capture 95% of the variance, reducing features to 190.

### Model Development
Initial model testing was performed to establish baselines:

| Model               | ROC-AUC Score |
|---------------------|---------------|
| Logistic Regression | 0.863         |
| Decision Tree       | 0.632         |
| XGBoost             | 0.853         |
| LightGBM            | 0.864         |

### Bayesian Optimization
Bayesian optimization is a strategy for finding the maximum value of an unknown function in as few iterations as possible. This technique builds a probabilistic model of the function and uses it to select the most promising hyperparameters to evaluate in the true objective function.

#### Why Bayesian Optimization?
- **Efficiency**: It is particularly useful when the function evaluations are expensive, like training a complex machine learning model.
- **Global Search**: Capable of finding a global optimum in the parameter space.
- **Less Trial and Error**: Uses prior knowledge to form a posterior that better guides the search.

### Hyperparameter Tuning with Optuna
Optuna was used to tune the LightGBM model's hyperparameters. This process aimed to improve the model's performance by optimizing its configuration.

#### Best Hyperparameters:
| Parameter          | Value     |
|--------------------|-----------|
| lambda_l1          | 0.039     |
| lambda_l2          | 0.004     |
| num_leaves         | 73        |
| feature_fraction   | 0.723     |
| bagging_fraction   | 0.411     |
| bagging_freq       | 1         |
| min_child_samples  | 86        |
| learning_rate      | 0.015     |
| n_estimators       | 1000      |

Best validation ROC-AUC: **0.873**

### Final Evaluation
The optimized LightGBM model demonstrated strong performance on the test set:

| Dataset | ROC-AUC Score |
|---------|---------------|
| Test    | 0.862         |


### Conclusion
The project developed a robust predictive model that efficiently handles high-dimensional data and predicts customer behavior with high accuracy. Future enhancements could explore more complex feature engineering, alternative machine learning techniques, and unsupervised learning methods.

### Future Work
- Exploring different ensemble techniques.
- Advanced feature engineering to uncover more complex patterns.
- Integrating unsupervised learning for feature discovery.
