#This file contains the steps taken to save the model for future use after optimal parameters were found from testing.
#No Testing on the test set is done or splitting the data into train and testing
#Import Statements
import math
import pickle
from os import path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

#The following library is where code for training was obtained and adapted from
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

#Probability threshold to classify a transaction
proba_threshold = 0.33

#load the credit card csv file
credit_data_df = pd.read_csv("data/dev_data.csv")
# create a dataframe of zeros   |
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

# count ones |
#no. of rows
numberOfOnes = credit_data_df_fraud.shape[0]

#LBR set to 1 | After testing this was found to be the best LB ratio for Random Forest
load_balancing_ratio = 1.0

#Set the number of zero's variable to be equal to the number of ones
numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)

#List of features used in training the model
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# choose a random sample of zeros (Legit Class) | random state used for the random state parameter to ensure the same data is used for all three models - deterministic
credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=23)

# merge the above with the ones (Fraud Class) and do the rest of the pipeline with it
result = credit_data_df_legit_random.append(credit_data_df_fraud)

# create dataframe X, which includes variables time, amount, V1, V2, V3, V4 etc
X = result[features]

# create array y, which includes the classification only
y = result['Class']

#Select the best features
select_kbest = SelectKBest(mutual_info_classif, k=26)
#Fit the method onto the data and then return a transformed array
X_new =select_kbest.fit_transform(X, y)
#Store the features selected
mask = select_kbest.get_support()
# use sklearns random forest to fit a model to train data
clf = RandomForestClassifier(n_estimators=100, random_state=23, class_weight='balanced')

#Train the model on ALL the data so that it can be saved and unused to test on unseen data
#Train the model using all the data, meaning learn about the relationship between feature and output class
clf.fit(X_new, y)

#Save the trained model together with the features selected
ml_object = [clf, mask]

#save the model for future use | Uncomment the line below if you would like to save the model
pickle.dump(ml_object, open(path.join('models', 'rf4.pkl'), 'wb'))
