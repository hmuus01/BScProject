#This file contains the steps taken to train the model and test the ML models with different SelectKBest functions
#on all the features and how this impacts the performance is analysed
#Import statements
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, f_regression,mutual_info_classif ,mutual_info_regression
import matplotlib.pyplot as plt

#The following library is where code for training was obtained and adapted from
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

#Probability threshold to classify a transaction
proba_threshold = 0.5

#load the credit card csv file
credit_data_df = pd.read_csv("../data/dev_data.csv")
# create a dataframe of zeros   |
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

# count ones |
#no. of rows
numberOfOnes = credit_data_df_fraud.shape[0]

#LBR set to 3 | After testing this was found to be the best LB ratio for Logistic Regression
load_balancing_ratio = 3.0

#Set the number of zero's variable to be equal to the number of ones
numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)

#A list of random seed's used for the random state parameter as way to counteract overfitting and get the mean
#of how the model performs from different data and different train-test splits
random_seeds = [12, 23, 34, 1, 56, 67, 45, 6]

#List of SelectkBest Functions | This is what is Tested In This File
algs = [f_regression, f_classif , mutual_info_regression, mutual_info_classif]

#Dictionary to store the mean accuracies of the optimizers
all_accuracys = {str(algs[0]):[], str(algs[1]):[], str(algs[2]):[], str(algs[3]):[]}
all_recalls = {str(algs[0]):[], str(algs[1]):[], str(algs[2]):[], str(algs[3]):[]}

#List of features used in training the model
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

#For each selectkbest function do the following:
for alg in algs:
    # Do the following as many times as the length of the feature list
    for k in range(1, len(features)):
        # Array to store the accuracies and the recalls
        accuracies = []
        recalls = []
        # Train & Test the following using a different random seed value
        for rs in random_seeds:
            # choose a random sample of zeros
            credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

            # merge the above with the ones and do the rest of the pipeline with it
            result = credit_data_df_legit_random.append(credit_data_df_fraud)

            # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 etc
            X = result[features]
            # create array y, which includes the classification only
            y = result['Class']

            # Select the best features Using the SelectKBest Method from sklearn
            select_kbest = SelectKBest(alg, k=k)
            # Fit the method onto the data and then return a transformed array
            X_new = select_kbest.fit_transform(X, y)

            # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y)

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
            #                                                    TRAINING ON THE TRAINING SET
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
            # use sklearns Logistic Regression to fit a model to train data
            clf = LogisticRegression(random_state=rs, solver='liblinear', class_weight='balanced')

            # Train the model using the training data, meaning learn about the relationship between feature and output class
            clf.fit(X_train, y_train)

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
            #                                                    TESTING ON THE TEST SET
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
            #     # for this classification use Predict_proba to give the probability of the classes whereas predict() just predicts the output class for the test set
            probs = clf.predict_proba(X_test)

            # store just the fraudulent class probabilities
            fraudulent_class_probabilities = probs[:, 1]

            # Classify whether a transaction is legit or fraud depending on if it above or below the threshold value
            y_pred = [1 if x >= proba_threshold else 0 for x in fraudulent_class_probabilities]

            # use sklearn metrics to judge accuracy of model using test data
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            print(acc)

            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
            # Legit - Class 0, Fraud - Class 1
            target_names = ['class 0', 'class 1']

            # Print the confusion matrix
            #print('Below is the confusion matrix')
            print(confusion_matrix(y_test, y_pred))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            print((tn, fp, fn, tp))

            # Calculate the recall using the formula and add it to the recalls array
            recall = tp / (tp + fn)
            recalls.append(recall)

            # Print the classification report
            print(classification_report(y_test, y_pred, target_names=target_names))


        # Calculate and Print the mean scores for the following metrics (below)
        mean_accuracy = np.mean(np.array(accuracies))
        mean_recall = np.mean(np.array(recalls))
        #Print the mean accuracy and recall scores
        print('accuracy mean = ' + str(mean_accuracy))
        print('recall mean = ' + str(mean_recall))

        #Store the mean accuracies and recalls for each selectkbest function in their respective array
        all_recalls[str(alg)].append(mean_recall)
        all_accuracys[str(alg)].append(mean_accuracy)


#Title of the plot
plt.title('SelectKbest Test on Features - Recalls')
#Plot the performance of the selectkbest function below on the recall
plt.plot(range(len(all_recalls[str(algs[0])])), all_recalls[str(algs[0])], label='f_regression')
#Plot the performance of the selectkbest function below on the recall
plt.plot(range(len(all_recalls[str(algs[1])])), all_recalls[str(algs[1])], label='f_classif')
#Plot the performance of the selectkbest function below on the recall
plt.plot(range(len(all_recalls[str(algs[2])])), all_recalls[str(algs[2])], label='mutual_info_regression')
#Plot the performance of the selectkbest function below on the recall
plt.plot(range(len(all_recalls[str(algs[3])])), all_recalls[str(algs[3])], label='mutual_info_classif')
#Label the y-axis
plt.ylabel('Recalls')
#Label the x-axis
plt.xlabel('Features')
#Legend to show which color represents which function
plt.legend()
#Display the graph
plt.show()

#Title of the plot
plt.title('SelectKbest Test on Features - Accuracies')
#Plot the performance of the selectkbest function below on the accuracy
plt.plot(range(len(all_accuracys[str(algs[0])])), all_accuracys[str(algs[0])], label='f_regression')
#Plot the performance of the selectkbest function below on the accuracy
plt.plot(range(len(all_accuracys[str(algs[1])])), all_accuracys[str(algs[1])], label='f_classif')
#Plot the performance of the selectkbest function below on the accuracy
plt.plot(range(len(all_accuracys[str(algs[2])])), all_accuracys[str(algs[2])], label='mutual_info_regression')
#Plot the performance of the selectkbest function below on the accuracy
plt.plot(range(len(all_accuracys[str(algs[3])])), all_accuracys[str(algs[3])], label='mutual_info_classif')
#Plot the y-axis
plt.ylabel('Accuracies')
#Plot the x-axis
plt.xlabel('Features')
#Legend to show which color represents which function
plt.legend()
#Display the graph
plt.show()
