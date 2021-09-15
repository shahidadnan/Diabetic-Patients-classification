#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X=pd.read_csv("K:/Data science/23.K-Nearest neighbour/Diabetes_XTrain.csv")
Y=pd.read_csv("K:/Data science/23.K-Nearest neighbour/Diabetes_YTrain.csv")
X=X.values
Y=Y.values


# euclidian distance

# In[7]:


def dist(x1,x2):
    return np.sqrt(sum(x1-x2)**2)


# In[8]:


def knn(X,Y,querypoint,k=5):
    m=X.shape[0]
    vals=[]
    for i in range(m):
        d=dist(querypoint,X[i])
        vals.append((d,Y[i]))
    
    vals=sorted(vals)
    vals=vals[:k]
    vals=np.array(vals)
    new_vals=np.unique(vals[:,1],return_counts=True)
    #print(new_vals[1].argmax())
    maxfreqindex=new_vals[1].argmax()
    #print(maxfreqindex)
    pred=new_vals[0][maxfreqindex]
    return pred


# In[10]:


X_test=pd.read_csv("K:/Data science/23.K-Nearest neighbour/Diabetes_XTest.csv")
X_test=X_test.values
m=X_test.shape[0]
Y_test=[]
for i in range(m):
    pred=knn(X,Y,X_test[i])
    Y_test.append(pred)
print(Y_test[:5])


# In[11]:


df=pd.DataFrame(data=Y_test,columns=["y"])
df.to_csv('diabetes_prediction.csv',index=False)
df.head()


# In[ ]:




