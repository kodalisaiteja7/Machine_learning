#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('50_Startups.csv')
print(df)


# In[3]:


df.dropna(axis=0,inplace=True)


# In[4]:


np.unique(df.duplicated())


# In[5]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df.State = le.fit_transform(df.State)
print(df.head())


# In[6]:


df.set_index('State')


# In[12]:


x = df.drop('Profit',axis = 1)
print(x)


# In[13]:


x.set_index('State')


# In[19]:


y = df['Profit']
print(y)


# In[22]:


y.rename(header='Profit')


# In[23]:


print(y.head())


# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


from sklearn import cross_validation


# In[29]:


x = np.array(x)


# In[30]:


y = np.array(y)


# In[31]:


print(x,y)


# In[33]:


x = preprocessing.scale(x)
print(x)


# In[34]:


x_train, x_test ,y_train, y_test = cross_validation.train_test_split(x,y, test_size = 0.2)


# In[42]:


clf = LinearRegression(n_jobs=10)


# In[53]:


clf.fit(x_train, y_train)


# In[54]:


accuracy = clf.score(x_test, y_test)


# In[55]:


print(accuracy)


# In[56]:


clf.predict(x_train)


# In[ ]:




