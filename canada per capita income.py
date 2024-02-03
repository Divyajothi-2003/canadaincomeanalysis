#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np


# In[56]:


logdf=pd.read_csv('canada_per_capita_income.csv')
lindf=pd.read_csv('canada_per_capita_income.csv')


# In[57]:


print(lindf.to_string())


# In[88]:


from matplotlib import pyplot as plt
plt.scatter(lindf['year'], lindf['per capita income (US$)'], marker='*', color='red')
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.show()


# In[61]:


from sklearn.linear_model import LinearRegression
X=lindf[['year']]
Y=lindf[['per capita income (US$)']]


# In[62]:


from sklearn.model_selection import train_test_split
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[63]:


model = LinearRegression()
model.fit(X_train_1, Y_train_1)


# In[64]:


y_pred = model.predict(X_test_1)
y_pred


# In[65]:


model.score(X,Y)


# In[66]:


from sklearn.linear_model import LogisticRegression
X=logdf[['year']]
y=logdf[['per capita income (US$)']]


# In[67]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[68]:


bins = [0, 15000, 30000, float('inf')]
labels = ['Low', 'Medium', 'High']
logdf['income_class'] = pd.cut(logdf['per capita income (US$)'], bins=bins, labels=labels)


# In[69]:


X = logdf[['year']]
y = logdf['income_class']


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[82]:


model1 = LogisticRegression()
model1.fit(X_train, y_train)


# In[83]:


y_pred1 = model.predict(X_test)
y_pred1
model.score(X,y)


# In[84]:


import sklearn
from sklearn.tree import DecisionTreeClassifier#scikit-learn


# In[85]:


X1 = logdf[['year']]
Y1= logdf['income_class']


# In[86]:


X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(X1, Y1, test_size=0.2, random_state=42)
model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train_2, Y_train_2)


# In[87]:


y_predict=model.predict(X_test_2)
y_predict
model2.score(X1,Y1)


# In[ ]:




