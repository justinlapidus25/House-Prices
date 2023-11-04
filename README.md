<p align="center">
  <img src="https://github.com/justinlapidus25/Housing-Pricing/assets/130884190/ca0cd798-f8c6-45db-a6aa-3e1ff2348bb2" width="800" alt="Your Image">
</p>

# House Prices
This repository contains code for predicting house prices using various machine-learning models. The goal of this project is to explore different regression algorithms, preprocess data, and evaluate model performance using Root Mean Squared Error (RMSE).

## Table of Contents
### Introduction
### Data Cleaning
### Encoding Categorical Features
### Data Splitting
### Machine Learning Models
### Feature Importance
### Conclusion

##Introduction
Predicting house prices is a common problem in the field of real estate and finance. Machine learning models can help in making accurate predictions based on various features of the houses. This project uses a dataset containing information about different houses, such as the number of bedrooms, square footage, neighborhood, and more, to predict their sale prices.

## Data Cleaning
Before building machine learning models, it's essential to clean the dataset. Data cleaning involves handling missing values, dropping redundant columns, and ensuring data consistency. In this project, we performed data cleaning to prepare the dataset for analysis.

Encoding Categorical Features
Categorical features, such as neighborhood or house style, need to be converted into a numerical format for machine learning algorithms. We used one-hot encoding to transform these features into binary columns, making them suitable for modeling.

## Data Splitting
To build and evaluate machine learning models, the dataset was split into input features (X) and the target variable (y). X contains the features used to make predictions, and y contains the sale prices we aim to predict.

## Machine Learning Models
This project explores various machine learning models to predict house prices. Different pipelines were created, including preprocessing steps such as scaling using StandardScaler, and models such as Linear Regression, Logistic Regression, and Random Forest Regressor were employed. The goal was to compare these models and identify the one with the lowest RMSE.

![sdaghakusdh](https://github.com/justinlapidus25/Housing-Pricing/assets/130884190/02486bb6-7445-4889-9550-c35be5daa9a2)
### Logistic Regression with overfitting 

![gadsgdg](https://github.com/justinlapidus25/Housing-Pricing/assets/130884190/082360ad-1956-4531-9cdd-effb7e3f57ea)

### Linear Regression with Grid search CV

![hhsjdfhkda](https://github.com/justinlapidus25/Housing-Pricing/assets/130884190/22a8dfd3-cc2c-4e2c-b8ab-644f623cfbc5)

### Random Forest Regression with Grid Search CV


## Feature Importance
Feature importance analysis using a Random Forest Regressor was performed to understand which features have the most significant impact on house prices. This analysis helps in feature selection and improving model accuracy.

![fkljdlfhlasd](https://github.com/justinlapidus25/Housing-Pricing/assets/130884190/12c06472-f520-4d6d-974b-a5c740de6bf7)

### The top 15 important Features Impacting Housing Sales

![fkljdlfhlasd](https://github.com/justinlapidus25/Housing-Pricing/assets/130884190/15c27f6e-cdfd-4487-90da-4173fc0c1a7f)

### The bottom 15 important Features Impacting Housing Sales


## Conclusion
### In conclusion, this code showcases the process of tackling a house price prediction problem by employing various machine learning models and data preprocessing techniques. It begins with data reading and cleaning, proceeds with encoding categorical features, and splits the data into input features and the target variable. Multiple machine learning pipelines are constructed, and each model's performance is evaluated using RMSE as a metric. Furthermore, feature importance is analyzed using a Random Forest Regressor to identify the top and bottom important features. While the code provides a solid foundation for solving the problem, there's still room for further exploration and model optimization to enhance predictive accuracy. To continue improving the model, one could experiment with different regression algorithms, fine-tune hyperparameters, consider feature engineering, and evaluate model performance on a separate test dataset. Additionally, cross-validation is essential for assessing the model's robustness and generalization to unseen data. In summary, this code demonstrates a structured approach to tackling a regression problem and lays the groundwork for future refinements and model selection.
