#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score


# In[35]:


url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset= pandas.read_csv (url, names=names)


# In[36]:


print(dataset.shape)


# In[37]:


print(dataset.head(30))


# In[38]:


dataset.describe()


# In[39]:


print(dataset.groupby('class').size())


# In[40]:


dataset.plot(kind='box' ,subplots= False ,layout=(2,2), sharex = False ,sharey = False)


# In[41]:


dataset.hist()


# In[42]:


scatter_matrix(dataset)


# In[43]:


array=dataset.values
X=array[:,:4]
Y=array[:,4]
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.2,random_state=6)


# In[50]:


models = []
models.append(('LR', LogisticRegression(max_iter=1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC ()))

results = []
names = []
for name, model in models:
    kfold=model_selection.KFold(n_splits=10, random_state=None ,shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg= "%s %f (%f)" %(name,cv_results.mean(),cv_results.std())
    print(msg)


# In[ ]:





# In[ ]:




