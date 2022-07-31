#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[2]:


path_ede1 = r"C:\Users\Dell\Desktop\REP\rep1.csv"
path_ede2 = r"C:\Users\Dell\Desktop\REP\rep2.csv"

columns = list(range(3, 11))
m1 = np.loadtxt(path_ede1, usecols=columns, dtype=np.float, delimiter=",")
m2 = np.loadtxt(path_ede2, usecols=columns, dtype=np.float, delimiter=",")
print(m1)


# In[3]:


title= plt.suptitle("Instances of tick bites for Ede (2006-2014)", size=20)
title.set_position([.5, 1.05])
e1 = m1[:, 0]
e2 = m2[:, 0]
xlinspace = np.linspace(0, len(e1) - 1, len(e1))
plt.plot(xlinspace, e1, "b-", label="Ede 1", linewidth=2)
plt.plot(xlinspace, e2, "r-", label="Ede 2", linewidth=2)
plt.ylabel("Tick activity", size=16)
plt.xlabel("Observation", size=16)
plt.legend()
plt.grid()
plt.show()


# In[4]:


labels = ['TMAX','TMIN','SD','PREC','VP','RH','EV']
title= plt.suptitle('Histogram of the weather variables in the Tick Bites dataset', size=20)
title.set_position([.5, 1.05])
plt.subplots_adjust(hspace=0.5, wspace=0.5)
for i in range(1, 8):
    current_feature = m1[:, i]
    plt.subplot(2, 4, i)
    plt.title(labels[i - 1], size=12)
    plt.hist(current_feature, bins=30, color="darkblue")
    plt.xlabel("Variable", size=10)
    plt.ylabel("Frequency", size=10)
    plt.grid()
    plt.tight_layout()
plt.show()


# In[5]:


# Separate the raw matrices into X and Y
xtrain = m1[:, 1:]
ytrain = m1[:, 0]
xtest = m2[:, 1:]
ytest = m2[:, 0]


# In[6]:


rf = RandomForestRegressor(n_estimators=100, criterion="mse", max_depth=None, max_features="auto", bootstrap=True,
                           min_samples_split=2, n_jobs=1)
rf.fit(xtrain, ytrain)
ypred = rf.predict(xtest)
print("RF: ", r2_score(ytest, ypred))

zipped_rf = list (zip (labels, rf.feature_importances_))
print(zipped_rf)


# In[7]:


# Apply GBR
gb = GradientBoostingRegressor(n_estimators=100, loss="ls", learning_rate=0.1, criterion="friedman_mse")
gb.fit(xtrain, ytrain)
ypred = gb.predict(xtest)
print("GB: ", r2_score(ytest, ypred))

zipped_gb = list (zip (labels, gb.feature_importances_))
print(zipped_gb)


# In[ ]:




