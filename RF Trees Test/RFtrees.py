#This file contains the steps taken to train the model and test the use of different
# number of trees (estimators) on the performance of the model
#Import statements
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt

#The following library is where code for training was obtained and adapted from
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

#Arrays to store the Accuracy, Recall and x-ticks
x_ticks=[]
k_accuracies=[]
k_recalls=[]

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

#LBR set to 1 | After testing this was found to be the best LB ratio for Random Forest
load_balancing_ratio = 1.0

#Set the number of zero's variable to be equal to the number of ones
numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)

#A list of random seed's used for the random state parameter as way to counteract overfitting and get the mean
#of how the model performs from different data and different train-test splits
random_seeds = [12, 23, 34, 1, 56, 67, 45, 6]

#List of features used in training the model
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Do the following in steps of 20, starting from 60 | Here i am testing different number of decision trees
for k in range(60, 180, 20):
    # Array to store the accuracies and the recalls
    accuracies = []
    recalls = []
    # Train & Test the following using a different random seed value
    for rs in random_seeds:
        # choose a random sample of zeros (Legit Class)
        credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

        # merge the above with the ones (Fraud Class) and do the rest of the pipeline with it
        result = credit_data_df_legit_random.append(credit_data_df_fraud)

        # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 etc
        X = result[features]

        # create array y, which includes the classification only
        y = result['Class']

        # Select the best features Using the SelectKBest Method from sklearn
        select_kbest = SelectKBest(mutual_info_classif, k=26)
        # Fit the method onto the data and then return a transformed array
        X_new =select_kbest.fit_transform(X, y)

        # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y)

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        #                                                    TRAINING ON THE TRAINING SET
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # use sklearns Random Forest to fit a model to train data
        clf = RandomForestClassifier(n_estimators=k, random_state=rs,class_weight='balanced')
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
        # print('Below is the confusion matrix')
        print(confusion_matrix(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print((tn, fp, fn, tp))

        # Calculate the recall using the formula and add it to the recalls array
        recall = tp / (tp + fn)
        recalls.append(recall)

        # Print the classification report
        print(classification_report(y_test, y_pred, target_names=target_names))

    # calculate the mean accuracy
    mean_accuracy = np.mean(np.array(accuracies))
    # Calculate the mean recall
    mean_recall = np.mean(np.array(recalls))
    # Print the calculates scores above
    print('accuracy mean = ' + str(mean_accuracy))
    print('recall mean = ' + str(mean_recall))

    # Append the mean accuracy scores to k accuracies
    k_accuracies.append(mean_accuracy)
    # Append the mean recall scores to k recalls
    k_recalls.append(mean_recall)
    # set the xticks to equal the number of trees tested
    x_ticks.append(k)


#Plot the accuracy scores
plt.plot(x_ticks, k_accuracies)
#Label the y-axis
plt.ylabel('Accuracies')
#Title of the plot
plt.title('Decision Trees Test - Accuracies')
#Set the xticks
plt.xticks(x_ticks)
#Label the x-axis
plt.xlabel('No. of Trees')
#Display the graph
plt.show()


#Plot the recall scores
plt.plot(x_ticks, k_recalls)
#Label the y-axis
plt.ylabel('Recalls')
#Set the xticks
plt.xticks(x_ticks)
#Title of the plot
plt.title('Decision Trees Test - Recalls')
#Label the x-axis
plt.xlabel('No. of Trees')
#Display the graph
plt.show()

