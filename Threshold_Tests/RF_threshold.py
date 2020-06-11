#This file contains the steps taken to train the model and test the performance of the 4 metrics
# precision, recall, f1-score and accuracy on different threshold values - range(0.0-0.95)
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

#Probability threshold(s) to classify a transaction | This is what is Tested in this file.
proba_threshold = np.arange(0.1, 1.0, 0.05)

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

#A list of random seed's used for the random state parameter as way to get the mean
#of how the model performs from different data and different train-test splits
random_seeds = [12, 23, 34, 1, 56, 67, 45, 6]

#List of features used in training the model
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Array to store the mean accuracies, recalls, precision and f1_score results for each probability threshold (pt)
recals_pt=[]
prec_pt = []
f1_pt = []
acc_pt = []
#For each probability threshold do the following:
for pt in proba_threshold:
    # Store the result of each probability threshold
    accuracies = []
    recalls = []
    precisions = []
    f1_scores = []
    #Train & Test the following using a different random seed value
    for rs in random_seeds:
        # choose a random sample of zeros (Legit Class)
        credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

        # merge the above with the ones (Fraud Class) and do the rest of the pipeline with it
        result = credit_data_df_legit_random.append(credit_data_df_fraud)

        # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 etc
        X = result[features]

        # create array y, which includes the classification only
        y = result['Class']

        # Select the best features | After Testing this was found to be the best amount of features for Random Forest
        select_kbest = SelectKBest(mutual_info_classif, k=26)
        # Fit the method onto the data and then return a transformed array
        X_new = select_kbest.fit_transform(X, y)

        # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y)

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
        #     # for this classification use Predict_proba to give the probability of the classes whereas predict() just predicts the output class for the test set
        probs = clf.predict_proba(X_test)

        # store just the fraudulent class probabilities
        fraudulent_class_probabilities = probs[:, 1]

        # Classify whether a transaction is legit or fraud depending on if it above or below the threshold value
        y_pred = [1 if x >= pt else 0 for x in fraudulent_class_probabilities]

        # use sklearn metrics to judge accuracy of model using test data
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)


        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
        # Legit - Class 0, Fraud - Class 1
        target_names = ['class 0', 'class 1']

        # Print the confusion matrix
        print('Below is the confusion matrix')
        print(confusion_matrix(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print((tn, fp, fn, tp))

        # Calculate the recall using the formula and add it to the recalls array
        recall = tp / (tp + fn)
        recalls.append(recall)

        # Calculate the precision using the formula and add it to the precisions array
        precision = tp / (tp + fp)
        precisions.append(precision)

        # Calculate the f1_score using the formula and add it to the f1 scores array
        f1_score = 2 * ((precision * recall) / (precision + recall))
        f1_scores.append(f1_score)

        # Print the classification report
        print(classification_report(y_test, y_pred, target_names=target_names))

    # Append all scores to their respective arrays in order to get the mean from the 8 random seeds at each threshold
    acc_pt.append(np.mean(np.array(accuracies)))
    recals_pt.append(np.mean(np.array(recalls)))
    prec_pt.append(np.mean(np.array(precisions)))
    f1_pt.append(np.mean(np.array(f1_scores)))
    print("f1 pt array " + str(f1_pt))

#Using Matplotlib, plot the following scores for at each classification threshold on the same graph
fig, ax = plt.subplots()
#Title of the plot
plt.title('RF - Threshold Tests')
#Plot the Recall
plt.plot(proba_threshold, recals_pt, 'r', label='Recall')
#Plot the Precision
plt.plot(proba_threshold, prec_pt, 'g',label='Precision')
#Plot the F1
plt.plot(proba_threshold, f1_pt, 'b',label='F1-Score')
#Manually set the xticks
ax.set_xticks(np.arange(0.1, 1.0, 0.05))
#Title of the y-axis
plt.ylabel('Scores')
#Title of x-axis
plt.xlabel('Thresholds')
#Legend to show which color represents which Metric
plt.legend(bbox_to_anchor=(1.0, 1.0))
#Add a grid to the plot to show the points better
plt.grid()
#Show the graph
plt.show()

############################################################################
#Calculate and print the mean score for all metrics
############################################################################
#calculate the mean accuracy
mean_prec = np.mean(np.array(prec_pt))
mean_accuracy = np.mean(np.array(acc_pt))
#Calculate the mean recall
mean_recall = np.mean(np.array(recals_pt))
# print("f1 is" +str(f1_scores))
mean_f1 = np.mean(np.array(f1_pt))
print('accuracy mean = ' + str(mean_accuracy))
print('recall mean = ' + str(mean_recall))
print('prec mean = ' + str(mean_prec))
print('f1 mean = ' + str(mean_f1))
############################################################################


