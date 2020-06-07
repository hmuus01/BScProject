#This file contains the steps taken to train the model and test the performance of the model
#using 20 different lBR values and analysing the performance
#Import Statements
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#The following library is where code for training was obtained and adapted from
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

#Array to store the mean accuracies and recalls
k_accuracies=[]
k_recalls=[]

#Probability threshold to classify a transaction
proba_threshold = 0.5

#load the credit card csv file
credit_data_df = pd.read_csv("../data/dev_data.csv")

# create a dataframe of zeros   | example rslt_df = dataframe[dataframe['Percentage'] > 80]
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

#A list of random seed's used for the random state parameter as way to counteract overfitting and get the mean
#of how the model performs from different data and different train-test splits
random_seeds = [12, 23, 34, 1, 56, 67, 45, 6]

#Load balancing range
lb_range=range(1, 21)

#List of features used in training the model
feature_headers = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

#Do all the steps below for each load balancing ratio
for load_balancing_ratio in lb_range:
    #Store the accuracy and recall scores
    accuracies = []
    recalls = []
    # **load-balancing** --> Class distribution
    numberOfOnes = credit_data_df_fraud.shape[0]
    numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)
    #For each random seed do the steps below (Training and evaluating)
    for rs in random_seeds:
        # choose a random sample of zeros (Legit Class)
        credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

        # merge the above with the ones (Fraud Class) and do the rest of the pipeline with it
        result = credit_data_df_legit_random.append(credit_data_df_fraud)

        # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 etc
        X = result[feature_headers]

        # create array y, which includes the classification only
        y = result['Class']

        # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=y)

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        #                                                    TRAINING ON THE TRAINING SET
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # use sklearns random forest to fit a model to train data
        clf = RandomForestClassifier(n_estimators=100, random_state=rs, class_weight='balanced')

        # Train the model using the training data, meaning learn about the relationship between feature and output class
        clf.fit(X_train, y_train)

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        #                                                    TESTING ON THE TEST SET
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # for this classification use Predict_proba to give the probability of the classes whereas predict() just predicts the output class for the test set
        probs = clf.predict_proba(X_test)

        # store just the fraudulent class probabilities
        fraudulent_class_probabilities = probs[:, 1]

        # Classify whether a transaction is legit or fraud depending on if it above or below the threshold value
        y_pred = [1 if x >= proba_threshold else 0 for x in fraudulent_class_probabilities]

        # use sklearn metrics to judge accuracy of model using test data
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # Print the output score
        print(acc)

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
        target_names = ['class 0', 'class 1']

        # Print the confusion matrix
        print('Below is the confusion matrix')
        print(confusion_matrix(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print((tn, fp, fn, tp))

        # Calculate the recall using the formula and add it to the recalls array
        recall = tp / (tp + fn)
        recalls.append(recall)

        # Print the classification report
        print(classification_report(y_test, y_pred, target_names=target_names))


    #Calculate the mean accuracy
    mean_accuracy = np.mean(np.array(accuracies))
    #Calculate the mean recall
    mean_recall = np.mean(np.array(recalls))

    #Add the mean accuracy to the K accuracies array
    k_accuracies.append(mean_accuracy)
    # Add the mean recall to the K recalls array
    k_recalls.append(mean_recall)

    #Print the accuracy and the recall means
    print('accuracy mean = ' + str(mean_accuracy))
    print('recall mean = ' + str(mean_recall))


#Plot the accuracy scores of different LBR ratios
plt.plot(lb_range, k_accuracies)
#Label the y-axis
plt.ylabel('Accuracies')
#Label the x-axis
plt.xlabel('Load-Balancing Ratio')
#Title of the plot
plt.title('RF - Load-Balancing Test on Accuracies')
#Set the xticks
plt.xticks(lb_range)
#Display the graph
plt.show()

#Plot the recall scores of different LBR ratios
plt.plot(lb_range, k_recalls)
#Label the y-axis
plt.ylabel('Recalls')
#Title of the plot
plt.title('RF - Load-Balancing Test on Recalls')
#Set the xticks
plt.xticks(lb_range)
#Label the x-axis
plt.xlabel('Load-Balancing Ratio')
#Display the graph
plt.show()

