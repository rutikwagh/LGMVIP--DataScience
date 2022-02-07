#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification Project using Machine Learning

# Iris flower classification is a very popular machine learning project. 
# 
# The iris dataset contains three classes of flowers, Versicolor, Setosa, Virginica, and each class contains 4 features, ‘Sepal length’, ‘Sepal width’, ‘Petal length’, ‘Petal width’. 
# 
# The aim of the iris flower classification is to predict flowers based on their specific features.

# Steps to Classify Iris Flower:
# iris-flower
# 
# 1. Load the data
# 2. Analyze and visualize the dataset
# 3. Model training.
# 4. Model Evaluation.
# 5. Testing the model.

# # Step 1 – Load the data:
# 

# In[76]:


# DataFlair Iris Classification
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


# Load the data
df = pd.read_csv('iris.data', names=columns)


# In[54]:


df.head()#for showing first 5 records


# In[55]:


df.tail()#for showing last 5 records


# In[56]:


df.shape#for showing size of dataset


# In[57]:


df.isnull().sum()#for showing sum of null values


# In[58]:


df.dtypes #for showing datatypes of dataset


# In[59]:


print(df)#for showing actaul dataset


# # Step 2 – Analyze and visualize the dataset:
#  

# In[60]:


# Some basic statistical analysis about the data
df.describe()


# From this description, we can see all the descriptions about the data, like average length and width, minimum value, maximum value, the 25%, 50%, and 75% distribution value, etc.

# In[61]:


df.boxplot()


# In[62]:


df.hist()


# In[63]:


# Visualize the whole dataset
#To visualize the whole dataset we used the seaborn pair plot method. It plots the whole dataset’s information.
sns.pairplot(df, hue='Class_labels')


# From this visualization, we can tell that iris-setosa is well separated from the other two flowers.
# 
# And iris virginica is the longest flower and iris setosa is the shortest.

# Now let’s plot the average of each feature of each class..............

# In[64]:


# Seperate features and target  
data = df.values
X = data[:,0:4]
Y = data[:,4]


# Here we separated the features from the target value.
# 

# In[65]:


# Calculate avarage of each features for all classes
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1]) for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25


# Np.average calculates the average from an array.
# 
# Here we used two for loops inside a list. This is known as list comprehension.
# 
# List comprehension helps to reduce the number of lines of code.
# 
# The Y_Data is a 1D array, but we have 4 features for every 3 classes. So we reshaped Y_Data to a (4, 3) shaped array.
# 
# Then we change the axis of the reshaped matrix.
# 

# In[66]:


# Plot the avarage
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()

#We used matplotlib to show the averages in a bar plot.


# Here we can clearly see the verginica is the longest and setosa is the shortest flower.

# # Step 3 – Model training:
# 

# In[67]:


# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# Using train_test_split we split the whole data into training and testing datasets. 
# 
# Later we’ll use the testing dataset to check the accuracy of the model.

# In[68]:


# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)


# Here we imported a support vector classifier from the scikit-learn support vector machine.
# 
# Then, we created an object and named it svn.
# 
# After that, we feed the training dataset into the algorithm by using the svn.fit() method.
# 

# # Step 4 – Model Evaluation:

# In[79]:


# Predict from the test dataset
predictions = svn.predict(X_test)


# Now we predict the classes from the test dataset using our trained model.
# 
# Then we check the accuracy score of the predicted classes.
# 
# accuracy_score() takes true values and predicted values and returns the percentage of accuracy.

# In[70]:


# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# The accuracy is above 96%.

# Now let’s see the detailed classification report based on the test dataset.

# In[71]:


# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# The classification report gives a detailed report of the prediction.
# 
# Precision defines the ratio of true positives to the sum of true positive and false positives.
# 
# Recall defines the ratio of true positive to the sum of true positive and false negative.
# 
# F1-score is the mean of precision and recall value.
# 
# Support is the number of actual occurrences of the class in the specified dataset.
# 

# # Step 5 – Testing the model:

# Here we take some random values based on the average plot to see if the model can predict accurately.

# In[72]:


X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))


# It looks like the model is predicting correctly because the setosa is shortest and virginica is the longest and versicolor is in between these two.

# In[73]:


# Save the model
import pickle
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)


# In[74]:


# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)


# In[75]:


model.predict(X_new)


# We can save the model using pickle format.
# 
# And again we can load the model in any other program using pickle and use it using model.predict to predict the iris data.
# 

# # Summary:
# 

# In this project, I have learned to train our own supervised machine learning model using Iris Flower Classification Project with Machine Learning. Through this project, I learned about machine learning, data analysis, data visualization, model creation, etc.
