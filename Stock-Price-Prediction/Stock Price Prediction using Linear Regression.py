#!/usr/bin/env python
# coding: utf-8

# In[50]:


import quandl
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


quandl.ApiConfig.api_key = '231xMTVqhSJJGRQCJasy'
df = quandl.get("EOD/AAPL",api_key='231xMTVqhSJJGRQCJasy')


# In[3]:


df.head()


# In[4]:


c=df.corr()
m=np.triu(np.ones_like(c,dtype=bool))
f, ax=plt.subplots(figsize=(6,5))
cmap=sns.diverging_palette(230,20,as_cmap=True)
sns.heatmap(c,mask=m,cmap=cmap,vmax=.3,center=0,square=True,linewidth=.3)


# In[5]:


#----------Making the data more meaningful-------------
df['HL_PCT']=((df['Adj_High']-df['Adj_Low'])/df['Adj_Close'])*100.0
df['PCT_change']=((df['Adj_Close']-df['Adj_Open'])/df['Adj_Open'])*100.0


# In[6]:


#-------_Removing unecessary columns-------------
df.drop(['Adj_Open','Adj_High','Adj_Low','Dividend','Split','Open','High','Low','Close','Volume'],axis=1,inplace=True)


# In[7]:


df.head()


# In[8]:


df.corr()


# In[9]:


fc='Adj_Close'
df.fillna(-9999,inplace=True)
fcout=int(math.ceil(0.01*len(df)))


# In[10]:


df.shape


# In[11]:


label=df["Adj_Close"].shift(-fcout)
print("Label total null values: ",label.isna().sum())
print("Label dimensions: ",label.shape)


# In[12]:


label.dropna(inplace=True)


# In[14]:


print(label)


# In[24]:


X=df[['Adj_Volume','HL_PCT','PCT_change']]
y=np.array(label)


# In[25]:


X_lately=X[-fcout:]
X=X[:-fcout]


# In[26]:


print(X.shape)
print(X_lately.shape)


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.33,random_state=6) 
norm=MinMaxScaler().fit(X_train) #----------Compute the minimum and maximum to be used for later scaling.-------


# In[82]:


X_train=norm.transform(X_train)
X_test=norm.transform(X_test)
X_lately=norm.transform(X_lately)


# In[83]:


reg=linear_model.LinearRegression()
reg.fit(X_train,y_train)


# In[84]:


reg.get_params()


# In[89]:


reg.coef_


# In[96]:


predictions=reg.predict(X_test)
pred_Xlate=reg.predict(X_lately)


# In[95]:


from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
rmse=sqrt(mean_squared_error(y_test,predictions))
print("RMSE:",rmse)
r2=r2_score(y_test,predictions)
print("R-Squared Error:",r2)
mse=mean_squared_error(y_test,predictions)
print("MSE:",mse)


# In[ ]:




