#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation
from sklearn.linear_model import LinearRegression


# In[3]:


df = pd.read_csv('USA_Housing.csv')
print(df.head())


# In[4]:


df.drop('Address',axis=1)


# In[5]:


df.set_index('Area Population')


# In[9]:


x = df.drop(['Price','Address'], axis=1)
print(x.head())


# In[11]:


y = df['Price']
print(y)


# In[12]:


x = np.array(x)


# In[13]:


y = np.array(y)


# In[14]:


clf = LinearRegression()


# In[17]:


x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.2)


# In[18]:


clf.fit(x_train,y_train)


# In[21]:


accuracy = clf.score(x_test,y_test)
print(accuracy)


# In[23]:


clf.predict(x_test)


# In[ ]:




