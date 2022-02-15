#!/usr/bin/env python
# coding: utf-8

# ### Decision Tree

# # Prediction using Decision Tree  Algorithm :

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


iris=pd.read_csv('Iris (1).csv')
iris


# In[3]:


iris.info()


# In[26]:


iris.head()


# In[27]:


iris.tail()


# In[28]:


iris.isnull().sum()


# In[4]:


iris.Species.value_counts()


# In[5]:


iris['Species_class']=np.where(iris.Species=='Iris-virginica',1,np.where(iris.Species=='Iris-versicolor',2,3))


# In[6]:


iris.Species_class.value_counts()


# In[7]:


iris.columns


# In[8]:


cols=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


# #### Model Preparation

# In[9]:


from sklearn.model_selection import train_test_split


train_X, test_X, train_y, test_y = train_test_split( iris[cols],
                                                  iris['Species_class'],
                                                  test_size = 0.2,
                                                  random_state = 123 )


# #### Model Building

# In[10]:


param_grid = {'max_depth': np.arange(2, 8),
             'max_features': np.arange(2,5)}


# In[12]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 10,verbose=1,n_jobs=-1)
tree.fit( train_X, train_y )


# In[13]:


tree.best_score_


# In[14]:


tree.best_estimator_


# In[15]:


tree.best_params_


# In[16]:


train_pred = tree.predict(train_X)


# In[17]:


test_pred = tree.predict(test_X)


# In[18]:


import sklearn.metrics as metrics
print(metrics.classification_report(test_y, test_pred))


# #### Building Final Decision Tree

# In[19]:


clf_tree = DecisionTreeClassifier( max_depth = 4, max_features=2)
clf_tree.fit( train_X, train_y )


# In[20]:


tree_test_pred = pd.DataFrame( { 'actual':  test_y,
                            'predicted': clf_tree.predict( test_X ) } )


# In[21]:


tree_test_pred.sample( n = 10 )


# In[22]:


metrics.accuracy_score( tree_test_pred.actual, tree_test_pred.predicted )


# In[24]:


tree_cm = metrics.confusion_matrix( tree_test_pred.predicted,
                                 tree_test_pred.actual)
sns.heatmap(tree_cm, annot=True,
         fmt='.2f',
         xticklabels = ["Yes", "No"] , yticklabels = ["Yes", "No"] )

plt.ylabel('True label')
plt.xlabel('Predicted label')


# #### Graphical Representation of Decision Tree

# In[25]:


from sklearn import tree
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,10), dpi=300)
tree.plot_tree(clf_tree,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')


# # THANK YOU !
