# Credit-Card-Default-Prediction

This project aims to build a classification model for predicting whether a credit card client will default on their payments. The dataset used for this project is the "Default of Credit Card Clients Dataset," which consists of 30,000 examples and 24 features. The goal is to estimate whether a person will default (fail to pay) their credit card bills, indicated by the "default.payment.next.month" column in the dataset.

## Problem Description

Credit card default prediction is a crucial task for financial institutions to assess the creditworthiness of their clients. By accurately predicting default risks, banks can take proactive measures to mitigate potential losses and make informed decisions regarding credit approvals.

## Dataset

The dataset used for this project is the [Default of Credit Card Clients Dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset). It contains the following information:

- **Features**: The dataset includes 24 features, such as payment history, demographic information, and bill statements, which can be utilized to predict credit card default.
- **Target Variable**: The target variable is the "default.payment.next.month" column, indicating whether a client will default on their credit card payment in the next month.

## Approach

To solve the credit card default prediction problem, we will employ various machine learning techniques and follow these steps:

1. **Data Preprocessing**: We will perform data cleaning, handle missing values, and preprocess the dataset for model training.
2. **Exploratory Data Analysis**: We will explore the dataset, visualize key features, and gain insights into the relationship between variables.
3. **Feature Engineering**: We will extract relevant features, perform feature selection if necessary, and transform the data for modeling.
4. **Model Development**: We will train different classification models, such as logistic regression, decision trees, random forests, or gradient boosting, to predict credit card default.
5. **Model Evaluation**: We will assess the performance of each model using appropriate evaluation metrics and select the best-performing model.
6. **Hyperparameter Tuning**: We will optimize the selected model's hyperparameters using techniques like grid search or Bayesian optimization to improve its performance.
7. **Final Model Selection**: We will choose the model with the highest predictive accuracy and deploy it for credit card default prediction.

## Associated Research Paper

For additional insights and comparison purposes, you can refer to the research paper titled "[Default of Credit Card Clients Dataset](https://www.sciencedirect.com/science/article/pii/S0957417407006719)". This research paper provides valuable background information and previous approaches to credit card default prediction.

## Usage

1. Clone this repository to your local machine.
2. Download the [Default of Credit Card Clients Dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset) and place it in the project directory.
3. Open the Jupyter Notebook and run the cells sequentially.
4. Follow the instructions provided in the notebook to preprocess the data, train models, and evaluate the results.

## Conclusion

Our best model was LightGBM classifier with tuned hyperparameters. It achieved cross-validation macro-average f1 score of 0.707. The scores do not seem to overfit much; the gap between mean train score (0.719) and mean cross-validation score (0.707) is not big. These scores are very similar to the tuned random forest. But random forest seems to overfit. Also, it's much slower than LightGBM. So picked LightGBM model as our final model. 

We observed the macro-average f1 score of 0.695 using this model on the held out test set, which is in line with mean cross-validation macro-average f1-score (0.707). So there doesn't seem to be severe optimization bias here.

We observed that L1 feature selection helped a tiny bit for random forests. But we did not observe any improvement in LightGBM scores with feature selection in the pipeline. In general, we have small number of features in this problem and feature selection doesn't seem crucial. 

Our analysis of feature importances shows that our `PAY_\d{0,2}`, `LIMIT_*`, and `PAY_AMT*` variables seems to be most important features. Although `SEX` feature doesn't show up as one of the most important features, depending upon the context it might be a good idea to drop this feature from our analysis. 
