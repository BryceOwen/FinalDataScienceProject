#!/usr/bin/env python
# coding: utf-8

# # Final Exam: Part B
# 
# * Section: Sec01
# 
# * Group Number: S1G1: Mercedes Bischoff, Ryan Mandel, Bryce Owen, Kimberly Tang
# 
# * Due Date: May 7, 2020
# 
# * Purpose: Culmination of Semester Knowledge Base

# ## Preprocessing data

# In[80]:


#--Load Libraries 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


# In[81]:


#--Import and Preview Data Set
plants = pd.read_excel('http://barney.gonzaga.edu/~chuang/data/plants.xlsx')
plants.head()


# In[82]:


#--Identify Missing Data
plants.isnull().sum()


# In[83]:


#--Identify records that have missing values
plants[plants.isnull().any(axis=1)]


# In[84]:


#Fill all missing data with averages grouped by species
plants['sepal_length'] = plants['sepal_length'].fillna(plants.groupby('species')['sepal_length'].transform('mean'))
plants['sepal_width'] = plants['sepal_width'].fillna(plants.groupby('species')['sepal_width'].transform('mean'))
plants['petal_length'] = plants['petal_length'].fillna(plants.groupby('species')['petal_length'].transform('mean'))
plants['petal_width'] = plants['petal_width'].fillna(plants.groupby('species')['petal_width'].transform('mean'))
plants.head()


# In[85]:


#-- Standardize numeric variable by adding zscore columns
plants['sepal_length_z'] = stats.zscore(plants['sepal_length'], nan_policy = 'omit')
plants['sepal_width_z'] = stats.zscore(plants['sepal_width'], nan_policy = 'omit')
plants['petal_length_z'] = stats.zscore(plants['petal_length'], nan_policy = 'omit')
plants['petal_width_z'] = stats.zscore(plants['petal_width'], nan_policy = 'omit')

plants.head()


# In[86]:


#--Identify rows with zscores >3 or <-3
plants[(plants['sepal_length_z'] > 3) | 
        (plants['sepal_length_z'] < -3) |
        (plants['sepal_width_z'] > 3) | 
        (plants['sepal_width_z'] < -3) |
        (plants['petal_length_z'] > 3) | 
        (plants['petal_length_z'] < -3) |  
        (plants['petal_width_z'] > 3) | 
        (plants['petal_width_z'] < -3)]


# In[87]:


#--Drop rows that conatin outliers
plants = plants.drop(plants[(plants['sepal_length_z'] > 3) | 
        (plants['sepal_length_z'] < -3) |
        (plants['sepal_width_z'] > 3) | 
        (plants['sepal_width_z'] < -3) |
        (plants['petal_length_z'] > 3) | 
        (plants['petal_length_z'] < -3) |  
        (plants['petal_width_z'] > 3) | 
        (plants['petal_width_z'] < -3)].index)
plants.shape


# In[88]:


#Find duplicate records
plants[plants.duplicated(subset=plants.columns.difference(['ID']))]


# In[89]:


#Drop those records
plants.drop_duplicates(subset=plants.columns.difference(['ID']), inplace = True)
plants.shape


# In[90]:


#--Add Species Name Column to Data Set
def Species(x):
    if x == 0:
        return "setosa"
    elif x == 1:
        return "versicolor"
    else:
        return "virginica"

plants['SpeciesName'] = plants['species'].apply(Species)
plants.head()


# ## Explore Dataset

# In[91]:


#--Histogram of Each Variable (Z score variables not needed because they are the same distribution as normal variables)
plants[['petal_length','petal_width','sepal_length','sepal_width','species']].hist(bins=10,figsize=(20,15))
plt.show()


# In[92]:


#--Correlation Chart (Z score variables not needed because they are the same distribution as normal variables)
plants[['petal_length','petal_width','sepal_length','sepal_width','species']].corr().style.background_gradient("Greens")


# ## Clustering

# In[93]:


#Choose variabes to cluster and justify choice
# We chose to select the two variables with the strongest correlation to species, petal_length and petal_width

X = plants[['petal_length', 'petal_width']]
X.head()


# In[94]:


#Plot the distribution of sepal_length and petal_length
plt.figure(figsize=(8,6))
sns.scatterplot(X['petal_length'], X['petal_width'])


# In[95]:


#Save means and standard deviations
#Used to standardize data of new plants for prediction
plants_mean = X.mean()
plants_std = X.std()
print(plants_mean)
print()
print(plants_std)


# In[96]:


#Standardize the data
z_score = stats.zscore(X)
X_z = pd.DataFrame(z_score, columns = ['petal_length_z', 'petal_width_z'])
X_z.tail()


# In[97]:


#Fit a model with the data
#Create three clusters
from sklearn.cluster import KMeans

kmeans_plants = KMeans(n_clusters = 3).fit(X_z)


# In[98]:


#Obtain the labels
cluster = kmeans_plants.labels_
cluster


# In[99]:


#Obtain the centroids
cluster_center = kmeans_plants.cluster_centers_
cluster_center


# In[100]:


X_z.tail()


# In[101]:


#Merge original data, rescaled data and add column to label the results of clustering
clt = pd.DataFrame(cluster, columns=['cluster'])
plants_cluster = X.merge(X_z, on = X.index)
plants_cluster = pd.concat([plants_cluster, clt], axis = 1, sort = True)
plants_cluster = plants_cluster.rename(columns={'key_0': 'ID'})
plants_cluster.tail()


# In[102]:


#Preview the new dataset
plants_cluster.head()


# In[103]:


#Plot clusters
plt.figure(figsize=(8,6))
sns.scatterplot(plants_cluster['petal_length_z'],
                plants_cluster['petal_width_z'],
                hue=plants_cluster['cluster'])

#Plot centroids of the clusters
plt.plot(cluster_center[:,0],
         cluster_center[:,1],
         'r*',
         markersize=10)
# Centroids marked with stars


# In[104]:


#Predict plants
#Suppose we have 5 plants and they have the following measurements
#The data is saved in a list, each element of which has two values: width and length
plant_list = [[4.5, 2.3],
            [2.92, 1.5],
             [1.2, 0.3],
             [1.2, 0.5],
             [4.9, 2]]
#Convert the list to a dataframe
newplant = pd.DataFrame(plant_list,columns=['petal_length','petal_width'])
newplant


# In[105]:


#Standardize the measurements of new plants
newplant_z = (newplant-plants_mean)/plants_std
newplant_z = newplant_z.rename(columns={'petal_length': 'petal_length_z', 'petal_width': 'petal_width_z'})
newplant_z


# In[106]:


#Prediction
preds = kmeans_plants.predict(newplant_z)
preds


# In[107]:


#Interpretations
#The first plant belongs to the cluster of high petal_length and petal_width
#The second plant belongs to the cluster of medium petal_length and petal_width
#The thrid plant belongs to the cluster of low petal_length and petal_width
#The fourth plant belongs to the cluster of low petal_length and petal_width
#The fifth plant belongs to the cluster of high petal_length and petal_width

# Combine original data, standardized data and predicted clusters
combined_newplant = pd.concat([newplant, newplant_z, pd.DataFrame(preds, columns=['cluster'])], axis = 1)
combined_newplant


# In[108]:


#Plot clusters
plt.figure(figsize=(8,6))
sns.scatterplot(plants_cluster['petal_length_z'],
                plants_cluster['petal_width_z'],
                hue=plants_cluster['cluster'])

#Plot centroids of the clusters
plt.plot(cluster_center[:,0],
         cluster_center[:,1],
         'r*',
         markersize=10)

#Plot predictions
plt.plot(newplant_z.iloc[:,0],
         newplant_z.iloc[:,1],
         'go',
         markersize=10)

# Graph shows where predicted flowers should lie with green dots


# ## Classification

# In[109]:


#--Preview Data and drop non-predictors
plants.drop(['ID','sepal_length_z','sepal_width_z','petal_length_z','petal_width_z','SpeciesName'], axis = 1, inplace = True)
plants.head()


# In[110]:


# establish predictors
outcome = 'species'
predictors = [c for c in plants.columns if c != outcome]
predictors


# In[111]:


# --Split training and testing
# Test size of 30% used which is around the industry standard
X = plants.drop('species',axis=1) # -- features --
y = plants['species']             # -- target --

x_train, x_test, y_train, y_test = train_test_split(X,y,
                                                    test_size = 0.3, 
                                                    random_state=1)


# In[112]:


# -- train neural nets and fit model--
# Solver type is lbfgs
# Hidden layer sizes is 3
ann_clf = MLPClassifier(hidden_layer_sizes = (3), 
                        activation = 'logistic', 
                        solver = 'lbfgs', random_state = 1)

ann_clf.fit(x_train,y_train)


# In[113]:


# -- confusion matrix --

metrics.confusion_matrix(y_true = y_train, 
                         y_pred = ann_clf.predict(x_train))


# In[114]:


# -- Use sklearn.metrics to present confustion_matrix --

# -- use seaborn heatmapt to present the confusion matrix --
# -- This is based on TRAINING DATA --
get_ipython().run_line_magic('matplotlib', 'inline')

sns.heatmap(metrics.confusion_matrix(y_true = y_train, 
                                     y_pred = ann_clf.predict(x_train)),
                                    annot=True,
                                    cbar=False,
                                    fmt='g',
                                    cmap = plt.cm.get_cmap('Blues'))
plt.ylabel('Actual')
plt.xlabel('Predicted')


# In[115]:


# -- Validation performance --
# -- Use test data --
metrics.confusion_matrix(y_true = y_test,
                         y_pred=ann_clf.predict(x_test))


# In[116]:


# --- Print Confusion Matrix Using Panda's crosstab function ---
sns.set_style('white')
print('--- Confusion Matrix (predict on rows, actual on columns ---')
cmtab = metrics.confusion_matrix(y_test,ann_clf.predict(x_test))

# confusion_matrix = pd.crosstab(preds, y_test,rownames=['Predicted'], colnames=['Actual'])
ax = sns.heatmap(cmtab, annot=True,fmt='g',cbar=False,cmap='Blues')
ax.set(xlabel="Predicted",ylabel="Actual");

# The confusion matrix shows that nearly all of the predicted values were correctly identified. There were 2 flowers 
# in classes 1 and 2 were misidentified. This is very minimal in the grand scheme of the dataset


# In[117]:


# show metrics
print(metrics.classification_report(y_true = y_test, 
                                    y_pred = ann_clf.predict(x_test)))

# The model has high performance in all areas. Especially the accuracy, precision, and F1 score which are 
# good metrics for this model because our biggest concern is classifying flowers correctly and avoiding false positives 

