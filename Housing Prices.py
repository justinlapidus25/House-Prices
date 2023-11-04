#!/usr/bin/env python
# coding: utf-8

# # Importing Packages

# In[163]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score




# In[ ]:





# # Data Reading and Cleaning

# In[123]:


Test= pd.read_csv('HousePrice/test.csv') #renaming pre split test data 


# In[124]:


Train= pd.read_csv('HousePrice/train.csv') #renaming pre split train data


# #### identifying redundant categories, or categories that only represent a small percentage of the houses included 

# In[125]:


columns_drop=['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','PoolQC','3SsnPorch','MiscFeature','Fence','FireplaceQu','GarageQual','GarageCond','BsmtFinSF2','BsmtUnfSF','HeatingQC','Electrical','BsmtFullBath','BsmtHalfBath']
columns_drop1=['LandContour','LandSlope','OverallQual','OverallCond','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond']
columns_drop2= ['MSSubClass','GarageType','GarageYrBlt','GarageFinish','LotFrontage','Alley','MSZoning','LotConfig','Condition1','Condition2','RoofStyle','RoofMatl','Foundation','BsmtQual','LowQualFinSF']


# In[126]:


columns_to_drop = columns_drop + columns_drop1+columns_drop2


# In[127]:


Updated_Test= Test.drop(columns=columns_to_drop)
Updated_Test.isna().sum() #Checking is there are any Nan values present 
Updated_Test.dropna() #dropping any Nan values that were present 


# In[128]:


Updated_Train= Train.drop(columns=columns_to_drop) #Checking if there are any Nan values present 
Updated_Train.isna().sum()


# In[129]:


Updated_Train.info() # checking the Dtype of the columns 


# # Encoding Categorical Columns 
# ### Since the categories represented are not ordinal I was able to use OneHotEncoder. This was necessary to change the data included to a variable not an object 
# 
# #### Encoding the Train Set

# In[130]:


categorical_columns = Updated_Train.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse=False)

encoder.fit(Updated_Train[categorical_columns])

encoded_columns = encoder.transform(Updated_Train[categorical_columns])

feature_names = encoder.get_feature_names_out(input_features=categorical_columns)

encoded_df = pd.DataFrame(encoded_columns, columns=feature_names)

Updated_Train.drop(categorical_columns, axis=1, inplace=True)

Updated_Train_encoded = pd.concat([Updated_Train, encoded_df], axis=1)


# #### Encoding Test set

# In[131]:


categorical_columns = Updated_Test.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse=False)

encoder.fit(Updated_Test[categorical_columns])

encoded_columns = encoder.transform(Updated_Test[categorical_columns])

feature_names = encoder.get_feature_names_out(input_features=categorical_columns)

encoded_df = pd.DataFrame(encoded_columns, columns=feature_names)

Updated_Test.drop(categorical_columns, axis=1, inplace=True)

Updated_Test_encoded = pd.concat([Updated_Train, encoded_df], axis=1)


# In[132]:


Updated_Train_encoded #checking the make sure all objects are now variables


# In[133]:


Updated_Test_encoded


# ### Creating X and Y variable (isolating sales price to predict on)

# In[134]:


X = Updated_Train_encoded.drop(columns='SalePrice')
y = Updated_Train_encoded['SalePrice']


# # Pipeline creation 
# ### Testing different machine learning models to find the lowest RMSE
# ### Scaling the data so all the variables are on the same scale. Using Standard Scaler within the pipelines

# ## Pipe Line One: Logistic Regression , Standard Scaler

# In[148]:


pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('model', LogisticRegression())
])

pipeline.fit(X, y)


# In[151]:


score = pipeline.score(X, y)

print("Logistic Regression Score:", score)


# In[154]:


pred = pipeline.predict(X)

plt.figure(figsize=(10, 7))
plt.scatter(y, pred, alpha=0.5, c='b', label='Predicted Values')  # Blue for predicted values
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Actual vs. Predicted Values')

plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2, label='Perfect Prediction Line')  # Red dashed line

plt.legend()
plt.show()







# ## Pipeline Two with Standard Scaler, Linear Regression, and GridSearchCV

# In[155]:


pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('model', LinearRegression())
])

param_grid = {
}

gridsearch = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  
    cv=5
)

gridsearch.fit(X, y)


# In[156]:


best_model = gridsearch.best_estimator_

best_params = gridsearch.best_params_


# In[157]:


best_score = gridsearch.best_score_
print(f"Best Mean Squared Error Score: {best_score}")


# In[158]:


pred = best_model.predict(X)

plt.figure(figsize=(10, 7))
plt.scatter(y, pred, alpha=0.5, c='b', label='Predicted Values')  # Blue for predicted values
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Actual vs. Predicted Values')

plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2, label='Perfect Prediction Line')  # Red dashed line

plt.legend()  

plt.show()


# ## Pipe Line: Random Forest, Standard Scaler, GridSearchCV

# In[143]:


pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('model', RandomForestRegressor(n_estimators=100, random_state=42)) 
])

param_grid = {
    'model__n_estimators': [100, 200, 300],  
    'model__max_depth': [None, 10, 20, 30], 
    'model__min_samples_split': [2, 5, 10] } 

gridsearch = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  
    cv=5  
)

gridsearch.fit(X, y)




# In[145]:


best_model = gridsearch.best_estimator_

best_params = gridsearch.best_params_


# In[146]:


best_score = gridsearch.best_score_
print(f"Best Mean Squared Error Score: {best_score}")


# In[176]:


pred = best_model.predict(X)

plt.figure(figsize=(10, 7))
plt.scatter(y, pred, alpha=0.5, c='b', label='Predicted Values')  # Blue for predicted values
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Actual vs. Predicted Values')

plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2, label='Perfect Prediction Line')  # Red dashed line

plt.legend()  

plt.show()


# In[177]:


from sklearn.ensemble import RandomForestRegressor

X = Updated_Train_scaled_df.drop("SalePrice", axis=1)  # Features (all columns except SalePrice)
y = Updated_Train_scaled_df["SalePrice"]  # Target variable

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X, y)

feature_importances = model.feature_importances_

importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

top_15_importance_df = importance_df.head(15)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=top_15_importance_df)
plt.title('Top 15 Feature Importance')
plt.show()





# In[178]:


low_15_importance_df = importance_df.tail(15)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=low_15_importance_df)
plt.title('Bottom 15 Feature Importance')
plt.show()


# # Conclusion 
# 
# ## In conclusion, this code showcases the process of tackling a house price prediction problem by employing various machine learning models and data preprocessing techniques. It begins with data reading and cleaning, proceeds with encoding categorical features, and splits the data into input features and the target variable. Multiple machine learning pipelines are constructed, and each model's performance is evaluated using RMSE as a metric.
# 
# ## Furthermore, feature importance is analyzed using a Random Forest Regressor to identify the top and bottom important features. While the code provides a solid foundation for solving the problem, there's still room for further exploration and model optimization to enhance predictive accuracy.
# 
# ## To continue improving the model, one could experiment with different regression algorithms, fine-tune hyperparameters, consider feature engineering, and evaluate model performance on a separate test dataset. Additionally, cross-validation is essential for assessing the model's robustness and generalization to unseen data.
# 
# ## In summary, this code demonstrates a structured approach to tackling a regression problem and lays the groundwork for future refinements and model selection.
