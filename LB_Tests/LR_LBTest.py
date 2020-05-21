#Import Statements
import math

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

import matplotlib.pyplot as plt

#Probability threshold to classify a transaction
proba_threshold = 0.5

#Load the credit card csv file
credit_data_df = pd.read_csv("../data/dev_data.csv")

#create a dataframe of zeros
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

#A list of random seed's used for the random state parameter as way to counteract overfitting and get the mean
#of how the model performs from different data and different train-test splits
random_seeds = [12, 23, 34, 1, 56, 67, 45, 6]

#Dictionary to store the mean accuracies of the optimizers
all_accuracys={'lbfgs':[], 'newton-cg':[], 'liblinear':[]}
all_recalls = {'lbfgs':[], 'newton-cg':[], 'liblinear':[]}

#Load balancing range
lb_range=range(1,21)

#List of Logistic Regression Optimizers used in this experiment
optimizers=['lbfgs','newton-cg','liblinear']

#List of features used in training the model
feature_headers = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

#For each optimizer do the following:
for optimizer in optimizers:
    #For each load balancing ratio do the following:
    for load_balancing_ratio in lb_range:
        # Array to store the accuracies and the recall results
        accuracies = []
        recalls = []
        # **load-balancing** --> Class Distribution
        numberOfOnes = credit_data_df_fraud.shape[0]
        numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)
        # For each random seed do the steps below (Training and evaluating)
        for rs in random_seeds:
            # choose a random sample of zeros (Legit Class)
            credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

            # merge the above with the ones (Fraud Class) and do the rest of the pipeline with it
            result = credit_data_df_legit_random.append(credit_data_df_fraud)

            # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 etc
            X = result[feature_headers]

            # create array y, which includes the classification only
            y = result['Class']

#################################################
            # NOT NEEDED FOR THIS TEST
            # # Select the best features
            # select_kbest = SelectKBest(f_classif, k=24)
            # # Fit the method onto the data and then return a transformed array
            # X_new = select_kbest.fit_transform(X, y)
            # # Store the features selected
            # mask = select_kbest.get_support()
#################################################

            # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=y)

            # use sklearns Logistic Regression to fit a model to train data
            clf = LogisticRegression(random_state=rs, solver=optimizer, class_weight='balanced', max_iter=1000)

            # Train the model using the training data, meaning learn about the relationship between feature and output class
            clf.fit(X_train, y_train)

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

            # Plot the points for the Roc curve and the auc score
            fpr, tpr, threshold = metrics.roc_curve(y_test, fraudulent_class_probabilities)
            roc_auc = metrics.auc(fpr, tpr)

            # Store the probability of a fraud transaction and the predictions and actutal class into a dataframe
            observations_df = pd.DataFrame(columns=['y_true', 'prediction', 'proba'])
            observations_df['y_true'] = y_test
            observations_df['prediction'] = y_pred
            observations_df['proba'] = fraudulent_class_probabilities
        #Calculate the mean accuracy for each run with the different random seeds
        mean_accuracy = np.mean(np.array(accuracies))

        #Calculate the mean recall for each run with the different random seeds
        mean_recall = np.mean(np.array(recalls))

        #Store the mean accuracies and recalls for each optimizer in their respective array
        all_recalls[optimizer].append(mean_recall)
        all_accuracys[optimizer].append(mean_accuracy)

        #Print the accuracy and recall mean
        print('accuracy mean = ' + str(mean_accuracy))
        print('recall mean = ' + str(mean_recall))

#Print the results
plt.title('Load-Balancing Test on Recalls')
plt.plot(lb_range, all_recalls['lbfgs'], label='lbfgs')
plt.plot(lb_range, all_recalls['newton-cg'], label='newton-cg')
#plt.plot(range(len(all_recalls['sag'])), all_recalls['sag'], label='sag')
plt.plot(lb_range, all_recalls['liblinear'], label='liblinear')
plt.ylabel('Recalls')
plt.xlabel('Load-Balancing Ratio')
plt.legend()
#plt.grid()
plt.show()

plt.title('Load-Balancing on Accuracies')
plt.plot(lb_range, all_accuracys['lbfgs'], label='lbfgs')
plt.plot(lb_range, all_accuracys['newton-cg'], label='newton-cg')
#plt.plot(range(len(all_accuracys['sag'])), all_accuracys['sag'], label='sag')
plt.plot(lb_range, all_accuracys['liblinear'], label='liblinear')
plt.ylabel('Accuracies')
plt.xlabel('Load-Balancing Ratio')
plt.legend()
#plt.grid()
plt.show()
