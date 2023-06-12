#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/LinlinDu/Data/main/cardio_train.csv')


# In[2]:


import capstone_group1 as cg


# In[3]:


data.describe()


# In[4]:


data.head()


# In[5]:


data.isna().sum()


# In[10]:


# there are 70000 different values in id column. column id is just index of rows. We can remove the column.
data['id'].value_counts()


# In[4]:


# Remove id
data = data.iloc[:,1:]


# In[5]:


# number of duplicated rows
print(data.duplicated().sum())


# In[24]:


data.shape


# In[13]:


data.drop_duplicates(keep='last',inplace=True)


# In[14]:


data.shape


# In[6]:


data['age'] = data['age']/365


# In[7]:


cg.remove_outliers(data,'ap_hi')


# In[8]:


data.isna().sum()


# In[9]:


cg.remove_outliers(data,'ap_lo')


# In[10]:


data.isna().sum()


# In[12]:


cg.remove_outliers(data,'height')


# In[13]:


data.isna().sum()


# In[14]:


cg.remove_outliers(data,'weight')


# In[15]:


data.isna().sum()


# In[16]:


data.dropna(inplace=True)


# In[17]:


data.isna().sum()


# In[38]:


data[data['ap_hi']<data['ap_lo']]
data=data[~(data['ap_hi']<data['ap_lo'])]


# In[39]:


data['BMI']=round(data['weight']/((data['height']/100)**2),2)


# In[40]:


data.head()


# In[41]:


sns.barplot(x='gender',y='BMI',data=data)
plt.show()


# In[42]:


data_categorical = data.loc[:,['cholesterol','gluc', 'smoke', 'alco', 'active']]
sns.countplot(x="variable", hue="value",data= pd.melt(data_categorical));


# In[43]:


df_long = pd.melt(data, id_vars=['cardio'], value_vars=['cholesterol','gluc', 'smoke', 'alco', 'active'])
sns.catplot(x="variable", hue="value", col="cardio",
                data=df_long, kind="count");


# In[44]:


sns.catplot(x="gender", y="BMI", hue="alco", col="cardio", data=data, color = "blue",kind="box", height=10, aspect=.7);


# In[45]:


corr = data.corr()
corr.style.background_gradient(cmap='RdYlGn')


# # modeling

# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[55]:


# spilit data: train, test
dataX = data.drop(['cardio'], axis=1)
dataY = data['cardio']
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.3, random_state=100)


# In[58]:


# LASSO and ridge regression
# get scores
l1Train, l1Test = Regression_scores(trainX,trainY,testX,testY,"l1",0.1,1.5,10)
l2Train, l2Test = Regression_scores(trainX,trainY,testX,testY,"l2",0.1,1.5,10)


# In[65]:


plt.plot(np.linspace(0.1,1.5,10),l1Train,'-g',label='L1Train')
plt.plot(np.linspace(0.1,1.5,10),l1Test,'--g',label='L1Test')
plt.plot(np.linspace(0.1,1.5,10),l2Train,'-r',label='L2Train')
plt.plot(np.linspace(0.1,1.5,10),l2Test,'--r',label='L2Test')
plt.legend(loc=4)
plt.show()


# In[66]:


# choose best parameter
modelL1 = LogisticRegression(penalty="l1",solver="liblinear",C=0.7,max_iter=5000)
modelL1 = modelL1.fit(trainX,trainY)
modelL2 = LogisticRegression(penalty="l2",solver="liblinear",C=0.7,max_iter=5000)
modelL2 = modelL2.fit(trainX,trainY)


# In[67]:


# show coefficient
modelL1.coef_


# In[68]:


modelL2.coef_


# In[69]:


print(accuracy_score(modelL1.predict(trainX),trainY),accuracy_score(modelL1.predict(testX),testY))


# In[70]:


print(accuracy_score(modelL2.predict(trainX),trainY),accuracy_score(modelL2.predict(testX),testY))


# In[89]:


modelL1.predict_proba(testX)


# In[87]:


modelL2.predict_proba(testX)


# In[77]:


# Decision Tree
modelDT = DecisionTreeClassifier(criterion="entropy",random_state=100)
modelDT = modelDT.fit(trainX, trainY)
print(modelDT.score(trainX, trainY),modelDT.score(testX, testY))


# In[78]:


# get scores
scoreTrain,scoreTest = cg.DecisionTree_scores(trainX,trainY,testX,testY,1,11)


# In[79]:


plt.plot(range(1,11),scoreTrain,color="red",label="train")
plt.plot(range(1,11),scoreTest,color="blue",label="test")
plt.xticks(range(1,11))
plt.legend()
plt.show()


# In[80]:


# choose best parameter
modelDT = DecisionTreeClassifier(criterion="entropy",random_state=100 ,max_depth=5)
modelDT = modelDT.fit(trainX, trainY)
print(modelDT.score(trainX, trainY),modelDT.score(testX, testY))


# In[85]:


# check feature importances
[*zip(dataX.columns,modelDT.feature_importances_)]


# In[88]:


modelDT.predict_proba(testX)


# In[ ]:




