#!/usr/bin/env python
# coding: utf-8

# In[264]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


# In[265]:


df = pd.read_csv('canada_per_capita_income.csv')


# In[266]:


X_lin = df[['year']]
Y_lin = df['per capita income (US$)']
X_train_lin, X_test_lin, Y_train_lin, Y_test_lin = train_test_split(X_lin, Y_lin, test_size=0.2, random_state=42)
lin_model = LinearRegression()
lin_model.fit(X_train_lin, Y_train_lin)
lin_score = lin_model.score(X_test_lin, Y_test_lin)


# In[267]:


from matplotlib import pyplot as plt
plt.scatter(lindf['year'], lindf['per capita income (US$)'], marker='*', color='red')
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.show()


# In[268]:


bins = [0, 15000, 30000, float('inf')]
labels = ['Low', 'Medium', 'High']
df['income_class'] = pd.cut(df['per capita income (US$)'], bins=bins, labels=labels)


# In[269]:


X_cat = df[['year']]
Y_cat = df['income_class']
X_train_cat, X_test_cat, Y_train_cat, Y_test_cat = train_test_split(X_cat, Y_cat, test_size=0.2, random_state=42)


# In[270]:


log_model = LogisticRegression()
log_model.fit(X_train_cat, Y_train_cat)
log_score = log_model.score(X_test_cat, Y_test_cat)


# In[271]:


tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_cat, Y_train_cat)
tree_score = tree_model.score(X_test_cat, Y_test_cat)


# In[272]:


model_names = ['Linear Regression', 'Logistic Regression', 'Decision Tree']
model_scores = [lin_score, log_score, tree_score]
print("Linear score",lin_score)
print("Logistic score",log_score)
print("Decision tree score",tree_score)


# In[273]:


plt.bar(model_names, model_scores, color=['blue', 'orange', 'green'])
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.show()


# In[ ]:




