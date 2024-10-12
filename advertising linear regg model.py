#!/usr/bin/env python
# coding: utf-8

# In[112]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# In[16]:


advert = pd.read_csv("advertising.csv")
advert.head()


# In[17]:


advert.tail(7)


# In[18]:


advert.shape


# In[19]:


advert.info()


# In[20]:


advert.describe()


# In[21]:


#visualising the data for TV 
sns.regplot(x='TV',y='Sales',data=advert)
plt.show()


# In[22]:


#visualising the data for radio
sns.regplot(x='Radio',y='Sales',data=advert)
plt.show()


# In[23]:


#visualising the data for Newspaper
sns.regplot(x='Radio',y='Newspaper',data=advert)
plt.show()


# In[24]:


#visualising the plots together.

sns.pairplot(data = advert, 
            x_vars = ['TV', 'Radio', 'Newspaper'],
            y_vars = ['Sales'] )
plt.show()


# In[25]:


#seeing the correlation of the matrix
advert.corr()


# In[26]:


sns.heatmap(advert.corr(), cmap = 'coolwarm', annot = True)
plt.show()


# In[28]:


#create X and Y for the data set

X = advert['TV']
y = advert['Sales']


# In[31]:


#train-test Split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.70, random_state = 100)
X_train.shape


# In[32]:


X_test.shape


# In[33]:


y_train.shape


# In[34]:


y_test.shape


# In[35]:


X_train.head(10)


# In[36]:


#Training the model
#using statsmodel
#adding the constant which is the problem of the statsmodel library
X_train_sm = sm.add_constant(X_train)

X_train_sm.head()


# In[37]:


#fitting the model 

lr = sm.OLS(y_train,X_train_sm)
lr_model =lr.fit()
lr_model.params


# In[38]:


lr_model.summary()


# In[43]:


### Note the p value is really low so the model is confident that there is a serious relation between them.
### Coef are noteworthy here as well
### R-squared is 0.81 which is very high
### p(F-statistics) is low => so the fit is not by chance



# In[44]:


#creating a ytrain_pred 

y_train_pred = lr_model.predict(X_train_sm)
y_train_pred.head()


# In[45]:


#let's now check the model by plotting it on a chart

plt.scatter(X_train, y_train)
plt.plot(X_train, y_train_pred, 'r')
plt.show()


# In[46]:


#Residual Analysis

res = y_train-y_train_pred
res.head()


# In[49]:


sns.distplot(res)
plt.show()


# In[50]:


# A bell curve is seen so this looks good enough to verify the assumption of error equally distributed


# In[51]:


#trying to note any pattern if possible in the next step


# In[53]:


plt.scatter(X_train,res);


# In[60]:


#Testing on the test set.

X_test_sm = sm.add_constant(X_test)
y_test_pred = lr_model.predict(X_test_sm)


# In[61]:


y_test_pred.head()


# In[63]:


r2 = r2_score(y_true = y_test, y_pred = y_test_pred)
r2


# In[64]:


#r2 score seems to be within 5% so this looks really good


# In[65]:


mean_squared_error(y_true = y_test, y_pred = y_test_pred)


# In[66]:


#let's visualise

plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, 'r')
plt.show()


# In[67]:


#USING SKLEARN Package


# In[68]:


#train test split 
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.70,random_state=100)


# In[70]:


#creating an object of linear Regression
lm = LinearRegression()


# In[103]:


#reshaping due to the Sklearn nuances now
X_Train_lm = X_train.values.reshape(-1,1)
X_Test_lm = X_test.values.reshape(-1,1)


# In[104]:


#fitting the model now
lm.fit(X_Train_lm,y_Train_lm)


# In[105]:


#seeing the coefficient and intercept
print(lm.coef_)
print(lm.intercept_)


# In[106]:


#making predictions
y_train_pred = lm.predict(X_Train_lm)
y_test_pred = lm.predict(X_Test_lm)


# In[108]:


y_test.shape


# In[109]:


#evaluate the model now
print(r2_score(y_true = y_train, y_pred = y_train_pred))
print(r2_score(y_true = y_test, y_pred = y_test_pred))


# In[ ]:




