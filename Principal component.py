#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv(
    filepath_or_buffer=r'C:\Users\Dell\Desktop\ASSIGNMENT\FINAL\PCAA.csv',
    header=None,
    sep=',')

df.columns=['DDVEC','FHVEC','FG','FHX','FHXH','FHN','FHNH','FXX','FXXH','TG','TN','TNH','TX','TXH','T10N','T10NH','SQ','SP','Q','DR','RH','RHX','RHXH','PG','PX','PXH','PN','PNH','VVN','VVNH','VVX','VVXH','NG','UG','UX','UXH','UN','UNH','EV24']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()


# In[3]:


from sklearn.preprocessing import StandardScaler


# In[4]:


features = ['DDVEC','FHVEC','FG','FHX','FHXH','FHN','FHNH','FXX','FXXH','TG','TN','TNH','TX','TXH','T10N','T10NH','SQ','SP','Q','DR','RH','RHX','RHXH','PG','PX','PXH','PN','PNH','VVN','VVNH','VVX','VVXH','NG','UG','UX','UXH','UN','UNH','EV24']


# In[5]:


x = df.loc[:,features].values
X = StandardScaler().fit_transform(x)


# In[6]:


from sklearn.decomposition import PCA


# In[65]:


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)


# In[66]:


pca.explained_variance_ratio_.sum()


# In[67]:


comps = pd.DataFrame(pca.components_,columns = df.columns)
comps


# In[68]:


import seaborn as sb


# In[64]:


sb.heatmap(comps)


# In[ ]:




