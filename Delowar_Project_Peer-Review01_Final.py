#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/images/IDSNlogo.png" width="300" alt="cognitiveclass.ai logo"  />
# </center>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>
# 

# In this notebook we try to practice all the classification algorithms that we have learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Let's first load required libraries:
# 

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset
# 

# This dataset is about past loans. The **Loan_train.csv** data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# | -------------- | ------------------------------------------------------------------------------------- |
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |
# 

# Let's download the dataset
# 

# In[2]:


get_ipython().system('wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv')


# ### Load Data From CSV File
# 

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object
# 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 

# Let’s see how many of each class is in our data set
# 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection
# 

# Let's plot some columns to underestand data better:
# 

# In[7]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction
# 

# ### Let's look at the day of the week people get the loan
# 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4
# 

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values
# 

# Let's look at gender:
# 

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Let's convert male to 0 and female to 1:
# 

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding
# 
# #### How about education?
# 

# In[14]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Features before One Hot Encoding
# 

# In[15]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame
# 

# In[16]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature Selection
# 

# Let's define feature sets, X:
# 

# In[17]:


X = Feature
X[0:5]


# What are our lables?
# 

# In[18]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data
# 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split)
# 

# In[20]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification
# 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# 
# *   K Nearest Neighbor(KNN)
# *   Decision Tree
# *   Support Vector Machine
# *   Logistic Regression
# 
# \__ Notice:\__
# 
# *   You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# *   You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# *   You should include the code of the algorithm in the following cells.
# 

# # K Nearest Neighbor(KNN)
# 
# Notice: You should find the best k to build the model with the best accuracy.\
# **warning:** You should not use the **loan_test.csv** for finding the best k, however, you can split your train_loan.csv into train and test to find the best **k**.
# 

# In[21]:


from sklearn.model_selection import train_test_split
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X, y, test_size=0.2)
print ('Train set for KNN:', X_train_k.shape,  y_train_k.shape)
print ('Test set for KNN:', X_test_k.shape,  y_test_k.shape)


# In[107]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

Ks = 20 # number of KNN model
mean_acc = np.zeros(Ks-1)
std_acc = np.zeros(Ks-1)

# for loop to find the best k
for n in range(1,Ks):
    # train K-NN model and predict
    k_neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train_k, y_train_k)
    yhat_k = k_neigh.predict(X_test_k)
    
    # calculate mean accuracy of n-th KNN model
    mean_acc[n-1] = metrics.accuracy_score(y_test_k, yhat_k)
    # calculate standard deviation of accuracy
    std_acc[n-1] = np.std(yhat_k==y_test_k)/np.sqrt(yhat_k.shape[0])
    
# display all accuracy

# print best accuracy
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


# In[24]:


X_train_KNN = X
y_train_KNN = y
print("Training Dataset: ", X_train_KNN.shape, y_train_KNN.shape)


# # Decision Tree
# 

# In[85]:


X_train_DT = X
y_train_DT = y
print ('Train set for Decision Tree:', X_train_DT.shape,  y_train_DT.shape)


# In[86]:


from sklearn.tree import DecisionTreeClassifier
# generate decision tree classifier using criterion="entropy" and max_depth=4
loanTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)


# In[ ]:





# # Support Vector Machine
# 

# In[26]:


X_train_SVM = X
y_train_SVM = y
print ('Train set for SVM:', X_train_SVM.shape,  y_train_SVM.shape)


# In[27]:


from sklearn import svm
clf_SVM = svm.SVC(kernel='rbf')


# In[ ]:





# # Logistic Regression
# 

# In[28]:


X_train_LR = X
y_train_LR = y
print ('Train set for LR:', X_train_LR.shape,  y_train_LR.shape)


# In[29]:


from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression(C=0.1, solver='liblinear')


# In[ ]:





# # Model Evaluation using Test set
# 

# In[30]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:
# 

# In[31]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation
# 

# In[32]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# ### Pre-processing 

# In[33]:


# Convert to date time object
test_df['due_date'] = pd.to_datetime(df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])

# Convert the day of the week people get the loan
test_df['dayofweek'] = df['effective_date'].dt.dayofweek
# people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

# Let's convert male to 0 and female to 1
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

# feture before one-shot
test_df[['Principal','terms','age','Gender','education']].head()


# ### Feature selction

# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame

# In[34]:


Feature_test = test_df[['Principal','terms','age','Gender','weekend']]
Feature_test = pd.concat([Feature_test,pd.get_dummies(test_df['education'])], axis=1)
Feature_test.drop(['Master or Above'], axis = 1,inplace=True)
Feature_test.head()


# #### Define test feature set, U; and labels to v

# In[36]:


# Feature
U = Feature_test
# labels
v = test_df['loan_status'].values
v_test = v


# #### Normalize test data 

# In[57]:


U = preprocessing.StandardScaler().fit(U).transform(U)
U_test = U
U[0:5]


# In[58]:


# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')


# ### Evaluating KNearest Neighbors Model

# In[108]:



from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

# Prediction model
KNN_neigh = KNeighborsClassifier(n_neighbors = 11).fit(X_train_KNN, y_train_KNN)
vhat_KNN = KNN_neigh.predict(U_test)

# Jaccard Accuracy
print("Jaccard Accuracy: ", jaccard_score(v_test, vhat_KNN, pos_label='PAIDOFF'))
      
# f1-score
print("f1-score: ",f1_score(v_test, vhat_KNN, average='weighted'))

# Classification Report
print ("classification_report:\n", classification_report(v_test, vhat_KNN))


# ### Evaluating Decision Tree Model

# In[102]:


# Prediction model
loanTree.fit(X_train_DT, y_train_DT)
vhat_DT = loanTree.predict(U_test)

# Jaccard Accuracy
print("Jaccard Accuracy: ", jaccard_score(v_test, vhat_DT, pos_label='PAIDOFF'))

# f1-score
print("f1-score: ",f1_score(v_test, vhat_DT, average='weighted'))

# Classification Report
print("classification_report:\n", classification_report(v_test, vhat_DT))


# ### Evaluating Support Vector Machine

# In[103]:


# Prediction model
clf_SVM.fit(X_train_DT, y_train_DT)
vhat_SVM = clf_SVM.predict(U_test)

# Jaccard Accuracy
print("Jaccard Accuracy: ", jaccard_score(v_test, vhat_SVM, pos_label='PAIDOFF'))
# f1-score
print("f1-score: ",f1_score(v_test, vhat_SVM, average='weighted'))

# Classification Report
print("Classification Report:\n", classification_report(v_test, vhat_SVM))


# ### Evaluating Logistic Regression Model

# In[104]:


# Prediction Model
clf_LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train_LR,y_train_LR)
vhat_LR = clf_LR.predict(U_test)
vhat_LR_prob = clf_LR.predict_proba(U_test)

# Jaccard Accuracy
print("Jaccard Accuracy: ",jaccard_score(v_test, vhat_LR, pos_label="PAIDOFF"))

#f1-score
print("f1-score: ",f1_score(v_test, vhat_LR, average='weighted'))

# Classification Report
print("classification_report", classification_report(v_test, vhat_LR))


# In[105]:


# LogLoss
from sklearn.metrics import log_loss
print(log_loss(v_test, vhat_LR_prob))


# # Report
# 
# You should be able to report the accuracy of the built model using different evaluation metrics:
# 

# | Algorithm          | Jaccard  | F1-score  | LogLoss |
# | ------------------ | -------  | --------  | ------- |
# | KNN                | 0.68     | 0.63      | NA      |
# | Decision Tree      | 0.75     | 0.72      | NA      |
# | SVM                | 0.72     | 0.62      | NA      |
# | LogisticRegression | 0.74     | 0.63      | 0.59    |
# 

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By    | Change Description                                                             |
# | ----------------- | ------- | ------------- | ------------------------------------------------------------------------------ |
# | 2020-10-27        | 2.1     | Lakshmi Holla | Made changes in import statement due to updates in version of  sklearn library |
# | 2020-08-27        | 2.0     | Malika Singla | Added lab to GitLab                                                            |
# 
# <hr>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 
# <p>
# 
