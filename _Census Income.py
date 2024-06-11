#!/usr/bin/env python
# coding: utf-8

# In[ ]:


- Prerak Pandya


# ## Census-income

# In[87]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# In[88]:


df1 = pd.read_csv(r"C:\Users\Prerak\Desktop\Census Income\census-income (7).csv")
df2 = pd.read_csv(r"C:\Users\Prerak\Desktop\Census Income\adult.csv")
df3 = pd.read_csv(r"C:\Users\Prerak\Desktop\Census Income\popdata (3).csv")


# In[89]:


df2


# Check the data type 

# In[90]:


df2.dtypes


# Check rows and columns

# In[91]:


df2.shape


# Check Null Values 

# In[92]:


df2.isnull().sum()
# Here we dont see any null value here 


# In[93]:


df2.duplicated().sum()


# In[94]:


df2.drop_duplicates(inplace= True)


# In[95]:


df2.duplicated().sum()


#  Here we Drop Duplicate values 

# In[96]:


df2['occupation'].unique()


# In[97]:


df2.drop(df2[df2['occupation'] == "?"].index, inplace=True)


# Drop  '?' value in the data set

# In[98]:


df2['marital-status'].unique()


# In[99]:


df2['education'].unique()


# In[100]:


df2['relationship'].unique()


# In[101]:


df2['native-country'].unique()


# In[102]:


df2.drop(df2[df2['native-country'] == "?"].index, inplace=True)


# In[103]:


df2


# In[104]:


df2['educational-num'].unique()


# In[105]:


df2['hours-per-week'].unique()


# In[106]:


df2['income'].unique()


# ### 

# #### Data Visulaization 

# In[21]:


plt.figure(figsize=(18,10))
sns.histplot(df2['education'])
plt.xticks(rotation = 90)
plt.title('Education')


# In[22]:


plt.figure(figsize=(18,10))
sns.histplot(df2['income'])
plt.xticks(rotation = 90)
plt.title('Education')


# In[23]:


plt.figure(figsize=(18,10))
sns.histplot(df2['native-country'])
plt.xticks(rotation = 90)
plt.title('Country')


# In[24]:


plt.figure(figsize=(18,10))
sns.histplot(df2['gender'])
plt.xticks(rotation = 90)
plt.title('Country')


# In[25]:


plt.figure(figsize=(18,10))
sns.histplot(df2['workclass'])
plt.xticks(rotation = 90)
plt.title('Country')


# In[26]:


plt.figure(figsize=(18,10))
sns.histplot(df2['occupation'])
plt.xticks(rotation = 90)
plt.title('Country')


# In[27]:


df2


# In[28]:


plt.figure(figsize=(15,10))
sns.barplot(df2,x = 'workclass',y = 'educational-num')


# In[ ]:


sns.qq


# #### Data Modeling 

# In[29]:


from sklearn.preprocessing import LabelEncoder


# In[30]:


la  = LabelEncoder()


# In[35]:


for i in df2.columns:
    if df2[i].dtype == 'object':  
        df2[i] = la.fit_transform(df2[i])
       
        


# In[36]:


df2


# Data Convert to labar formate 

# In[39]:


x = df2.iloc[::,:-1] #here  x is  imdependent Colunms 
y = df2[['income']]# depended columns


# In[42]:


from sklearn.model_selection import train_test_split


# In[69]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)


# In[70]:


# So Our Dependedent variable is Binary  so we can use classification algorithm called Logistic Regression 
from sklearn.linear_model import LogisticRegression


# In[71]:


lo = LogisticRegression()


# In[72]:


#lets build the model 
model = lo.fit(x_train,y_train)                    


# In[73]:


#predict income 
Prediction = model.predict(x_test)


# In[74]:


# if we want to check our model is accurate or not we use accurancy score 
from sklearn.metrics import accuracy_score


# In[75]:


accuracy_score(Prediction,y_test)
#here we see our model accurancy is 78 % 


# In[76]:


# Now we can  use differnt model like Decision Tress,RandomForest
from sklearn.ensemble import RandomForestClassifier


# In[77]:


ra = RandomForestClassifier()


# In[78]:


model_random = ra.fit(x_train,y_train)


# In[79]:


prediction_random = model_random.predict(x_test)


# In[80]:


accuracy_score(prediction_random,y_test)


# In[81]:


from sklearn.tree import DecisionTreeClassifier


# In[82]:


de = DecisionTreeClassifier()


# In[83]:


model_decision_tree = de.fit(x_train,y_train)


# In[84]:


prediction_decision = model_decision_tree.predict(x_test)


# In[86]:


accuracy_score(prediction_decision,y_test)
# Here we random forest algorithm give us 81% accurancy 


# Conclusion - We used logistic regression,Decision Tree,randomforest algorithm for predict tareget variable  and we get 85 % accurancy by randomforest algorithm 

# In[ ]:




