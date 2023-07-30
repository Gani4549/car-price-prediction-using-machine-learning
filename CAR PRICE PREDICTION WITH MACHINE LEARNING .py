#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


df=pd.read_csv("carpricetrain.csv")


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[5]:


df.drop(['car_ID'],axis=1,inplace=True)


# In[6]:


df


# In[7]:


df.shape


# # Exploratory Data Analysis

# ## Numerical features

# In[8]:


numerical_features=[feature for feature in df.columns if df[feature].dtype!='O']


# In[9]:


numerical_features


# In[10]:


plt.subplots(figsize=(20,15))
sns.heatmap(df[numerical_features].corr(),annot=True,linewidth=1)


# ## Categorical features

# In[11]:


categorical_features=[feature for feature in df.columns if df[feature].dtype=='O']


# In[12]:


categorical_features


# In[13]:


len(numerical_features)+len(categorical_features)


# ## Discrete variables

# In[14]:


discrete_variables=[feature for feature in numerical_features if len(df[feature].unique())<25]


# In[15]:


discrete_variables


# ## continuous variables

# In[16]:


continuous_variables=[feature for feature in numerical_features if feature not in discrete_variables]


# In[17]:


continuous_variables


# In[18]:


for feature in continuous_variables:
    data=df.copy()
    data[feature].hist()
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.title(feature)
    plt.show()


# ## Outliers

# In[19]:


for feature in continuous_variables:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.show()


# In[20]:


#To find the unique values of a categorical features
for feature in categorical_features:
    print(feature,len(df[feature].unique()))


# In[21]:


for feature in categorical_features:
    data=df.copy()
    data.groupby(feature)['price'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.show()


# ## Feature Engineering

# In[22]:


#we don't have null values now we have to handle the cateogrical_features


# ## Handling Categorical features

# In[23]:


categorical_features


# In[24]:


df.drop(['CarName'],axis=1,inplace=True)


# In[25]:


df


# In[26]:


categorical_features.remove("CarName")
for feature in categorical_features:
    print(feature,len(df[feature].unique()),df[feature].unique())


# In[27]:


fueltype=df[['fueltype']]
fueltype=pd.get_dummies(fueltype,drop_first=True)


# In[28]:


fueltype


# In[29]:


aspiration=df[['aspiration']]
aspiration=pd.get_dummies(aspiration,drop_first=True)


# In[30]:


aspiration


# In[31]:


df['doornumber'].replace({'two':2,'four':4},inplace=True)


# In[32]:


df


# In[33]:


carbody=df[['carbody']]
carbody=pd.get_dummies(carbody,drop_first=True)


# In[34]:


carbody


# In[35]:


drivewheel=df[['drivewheel']]
drivewheel=pd.get_dummies(drivewheel,drop_first=True)


# In[36]:


drivewheel


# In[37]:


enginelocation=df[['enginelocation']]
enginelocation=pd.get_dummies(enginelocation,drop_first=True)


# In[38]:


enginelocation


# In[39]:


enginetype=df[['enginetype']]
enginetype=pd.get_dummies(enginetype,drop_first=True)


# In[40]:


enginetype


# In[41]:


df['cylindernumber'].replace({'two':2,'four':4,'six':6,'three':3,'five':5,'twelve':12,'eight':8},inplace=True)


# In[42]:


df['cylindernumber']


# In[43]:


fuelsystem=df[['fuelsystem']]
fuelsystem=pd.get_dummies(fuelsystem,drop_first=True)


# In[44]:


fuelsystem


# In[45]:


df=pd.concat([df,fueltype,aspiration,carbody,drivewheel,enginelocation,enginetype,fuelsystem],axis=1)


# In[46]:


df


# In[47]:


df.drop(['fueltype','aspiration','carbody','drivewheel','enginelocation','enginetype','fuelsystem'],axis=1,inplace=True)


# In[48]:


df


# In[49]:


df.columns


# In[50]:


X=df.drop(labels=['price'],axis=1)
y=df['price']


# In[51]:


X


# In[52]:


y


# ## Training

# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.3)


# In[55]:


print(X_train.shape,X_test.shape)


# In[56]:


from sklearn.ensemble import RandomForestRegressor


# In[57]:


rf=RandomForestRegressor()


# In[58]:


rf.fit(X_train,y_train)


# In[59]:


rf.score(X_train,y_train)


# In[60]:


rf.score(X_test,y_test)


# In[61]:


y_pred=rf.predict(X_test)


# In[62]:


from sklearn.metrics import classification_report, confusion_matrix


# In[63]:


from sklearn import metrics
import numpy as np
print('MAE : ',metrics.mean_absolute_error(y_test,y_pred))
print('MSE : ',metrics.mean_squared_error(y_test,y_pred))
print('RMSE : ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[64]:


metrics.r2_score(y_test,y_pred)


# In[ ]:




