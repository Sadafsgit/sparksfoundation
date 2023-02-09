#!/usr/bin/env python
# coding: utf-8

# ![sparks.jpg](attachment:sparks.jpg)

# ## Author: Sadaf Shaikh

# ### THE SPARKS FOUNDATION: DATA SCINCE AND BUSINESS ANALYTICS

# #### Task 1 - Prediction Using Supervised ML

# #### Problem Statement :
# #### What will be predicted score if a student studies for 9.25 hrs/ day?
# #### Dataset: http://bit.ly/w-data

# In[1]:


#Supress warnings
import warnings
warnings.filterwarnings('ignore')


# **IMPORTING LIBRARIES**

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **LOADING THE DATASET**

# In[3]:


data_df = pd.read_csv('http://bit.ly/w-data')
data_df.head()


# In[4]:


data_df.shape


#  As we can see that there are 25 rows and 2 columns in the dataset.

# **DATA EXPLORATION**

# In[5]:


data_df.info()


# In[6]:


data_df.describe()


# In[7]:


data_df.isnull().sum()


# There is `no null` values in the dataset.

# **DATA VISUALIZATIONS**

# Let's `plot` our data points on `2-D graph` to `visualize` our dataset and see if we can manually find any `relationship` between the `data`.

# In[8]:


sns.set_style('darkgrid')
plt.plot(data_df['Hours'],data_df['Scores'],'oc')
plt.xlabel('Hours',fontsize = 20)
plt.ylabel('Scores',fontsize = 20)
plt.title('Hours Vs. Scores',fontsize = 20)
plt.legend(['Scores']);


# From the graph, we can clearly see that there is a `positive linear relation` between the `number of hours` studied and `percentage of score`.

# In[9]:


sns.regplot(x = data_df['Hours'], y = data_df['Scores'])
plt.title('Regression Plot')
plt.xlabel('Hours studies')
plt.ylabel('Percentage');


# From the `regression plot` it is confirmed that the `parameters` are `positively` correlated

# **BOXPLOT OF DATASET**

# In[10]:


sns.boxplot(data=data_df[['Hours','Scores']]);


# From the above plot, we can clearly see that there is `no outliers` in the data.

# **SPLITTING THE DATASET**

# In[16]:


X = data_df.iloc[:,:-1].values
y = data_df.iloc[:,-1].values


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,random_state = 0)


# In[18]:


X_train.shape, X_test.shape 


# Here we can see that `80%` of the data is used for `training` and the rest `20%` is used for `testing`. 

# **TRAINING THE MODEL**

# In[19]:


from sklearn.linear_model import LinearRegression
lg = LinearRegression()
lg.fit(X_train,Y_train)


# In[20]:


line = lg.coef_ * X + lg.intercept_
plt.scatter(data_df['Hours'],data_df['Scores'])
plt.plot(X,line)
plt.xlabel('Hours')
plt.ylabel('Percentage');


# **MAKING PREDICATION**

# Now, we will `test` our `algorithm` with the rest `20%` of the data that we have `splitted` and make `predictions`.

# In[24]:


y_pred = lg.predict(X_test).round()
prediction = pd.DataFrame({'Hours': [i[0] for i in X_test] ,'Predicted Scores':[y for y in y_pred]})
prediction


# **COMPAIRING ACTUAL SCORE VS. PREDICATED SCORE**

# In[27]:


pred = pd.DataFrame({'Actual Score': Y_test, 'Predicted Score%':y_pred})
pred


# What will be predicted score if a student studies for 9.25 hrs/ day?

# In[26]:


hours = np.array([[9.25]])
pred = lg.predict(hours)
print('No. of hours = {}'.format(hours[0][0]))
print('Predicted Score = %.2f'%pred)


# According to the `regression` model if a student `studies` `9.25` hours a day, he/she is likely to `score` `93.89` marks

# **EVALUATING THE MODEL**

# In[25]:


from sklearn.metrics import mean_absolute_error
print('Mean absolute error = %.2f'%mean_absolute_error(Y_test,y_pred))


# Small value of `mean absolute error` states that the chances of error or wrong forecasting through the model are `very less`.

# **Thank You :)**
