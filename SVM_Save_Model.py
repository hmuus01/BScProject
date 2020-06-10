#This file contains the steps taken to save the model for future use after optimal parameters were found from testing.
#No Testing on the test set is done or splitting the data into train and testing
#Import Statements
import math
import pickle
from os import path
import pandas as pd
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif

#The following library is where code for training was obtained and adapted from
#https://scikit-learn.org/stable/modules/svm.html#classification

#Probability threshold to classify a transaction
proba_threshold = 0.24

#load the credit card csv file
credit_data_df = pd.read_csv("data/dev_data.csv")
# create a dataframe of zeros   |
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

# count ones |
#no. of rows
numberOfOnes = credit_data_df_fraud.shape[0]

#LBR set to 1 | After testing this was found to be the best LB ratio for Support Vector Machines
load_balancing_ratio = 1.0

#Set the number of zero's variable to be equal to the number of ones
numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)

#List of features used in training the model
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# choose a random sample of zeros (Legit Class)| random state used for the random state parameter to ensure the same data is used for all three models - deterministic
credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=23)

# merge the above with the ones (Fraud Class) and do the rest of the pipeline with it
result = credit_data_df_legit_random.append(credit_data_df_fraud)

# create dataframe X, which includes variables time, amount, V1, V2, V3, V4 etc
X = result[features]

# create array y, which includes the classification only
y = result['Class']

#Select the best features | After Testing this was found to be the best amount of features for Support Vector Machines
select_kbest = SelectKBest(f_classif, k=5)
#Fit the method onto the data and then return a transformed array
X_new =select_kbest.fit_transform(X, y)
#Store the features selected
features_saved = select_kbest.get_support()

# use sklearns Support Vector Machines to fit a model to train data
clf = svm.SVC(C=1, kernel='linear', probability=True, random_state=12, class_weight='balanced')

#Train the model using all the data, meaning learn about the relationship between feature and output class
#Train the model on ALL the data so that it can be saved and unused to test on unseen data
clf.fit(X_new,y)

#Save the trained model together with the features selected
model_and_features = [clf, features_saved]

#save the model for future use
pickle.dump(model_and_features, open(path.join('models', 'svm.pkl'), 'wb'))


