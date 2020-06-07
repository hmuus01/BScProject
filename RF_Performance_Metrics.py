#This file contains the steps taken to train the model and test the performance of the model
#using the 4 performance metrics in chapter 4.6 of the report
#Import statements
import math
import random
import pandas as pd
import numpy as np
from scipy.stats import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt

#The following library is where code for training was obtained and adapted from
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

#Probability threshold to classify a transaction
#proba_threshold = 0.33 (for the performace of random forest with a threshold of 0.33 Uncomment Line)
proba_threshold = 0.5

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

#A number(2000) of random seed's used for the random state parameter as way to counteract overfitting and get the mean
#of how the model performs from different data and different train-test splits
num_seeds=2000
random_seeds=[]
while len(random_seeds) < num_seeds:
    num = random.randint(0, 5*num_seeds)
    if num not in random_seeds:
        random_seeds.append(num)

# Method to plot the ROC curve
def plot_roc():
    plt.title('RF - Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

#List of features used in training the model
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Array to store the accuracies, recalls, precision and f1_score results
accuracies = []
recalls = []
precisions = []
f1_scores = []

#Train & Test the model using different random seeds
#Do the steps below for each random seed
for rs in random_seeds:
    print(rs)
    # choose a random sample of zeros (Legit Class)
    credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

    # merge the above with the ones (Fraud Class) and do the rest of the pipeline with it
    result = credit_data_df_legit_random.append(credit_data_df_fraud)

    # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 etc
    X = result[features]

    # create array y, which includes the classification only
    y = result['Class']

    #Select the best features | After Testing this was found to be the best amount of features for Random Forest
    select_kbest = SelectKBest(mutual_info_classif, k=26)
    #Fit the method onto the data and then return a transformed array
    X_new = select_kbest.fit_transform(X, y)

    # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    #                                                    TRAINING ON THE TRAINING SET
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # use sklearns random forest to fit a model to train data
    clf = RandomForestClassifier(n_estimators=100, random_state=rs, class_weight='balanced')

    #Train the model using the training data, meaning learn about the relationship between feature and output class
    clf.fit(X_train, y_train)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------#
                                            #TESTING ON THE TEST SET
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
     # for this classification use Predict_proba to give the probability of the classes whereas predict() just predicts the output class for the test set
    probs = clf.predict_proba(X_test)

    #store just the fraudulent class probabilities
    fraudulent_class_probabilities = probs[:, 1]

    #Classify whether a transaction is legit or fraud depending on if it above or below the threshold value
    y_pred = [1 if x >= proba_threshold else 0 for x in fraudulent_class_probabilities]

    # use sklearn metrics to judge accuracy of model using test data
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    #Print the accuracy score
    print(acc)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    # Legit - Class 0, Fraud - Class 1
    target_names = ['class 0', 'class 1']

    #Make the confusion matric using the predicted results and check whether its similar to the actual results
    cm = confusion_matrix(y_test,y_pred)

    #Print the confusion matrix
    print('Below is the confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print((tn, fp, fn, tp))

    #Calculate the recall using the formula and add it to the recalls array
    recall = tp / (tp + fn)
    recalls.append(recall)

    #Calculate the precision using the formula and add it to the precisions array
    precision = tp / (tp + fp)
    precisions.append(precision)

    #Calculate the f1_score using the formula and add it to the f1 scores array
    f1_score = 2 * ((precision * recall) / (precision + recall))
    f1_scores.append(f1_score)

    #Print the classification report
    print(classification_report(y_test, y_pred, target_names=target_names))

    #Plot the points for the Roc curve and the auc score
    fpr, tpr, threshold = metrics.roc_curve(y_test, fraudulent_class_probabilities)
    roc_auc = metrics.auc(fpr, tpr)

#Display the Roc - (Uncomment to display)
#plot_roc()

#calculate the mean accuracy and recall from the respective arrays
mean_accuracy = np.mean(np.array(accuracies))
mean_recall = np.mean(np.array(recalls))

#Print the mean score for all metrics
print('accuracy mean = ' + str(mean_accuracy))
print('recall mean = ' + str(mean_recall))
print('precision mean = ' + str(np.mean(np.array(precisions))))
print('F1 mean = ' + str(np.mean(np.array(f1_scores))))

print('rf recall trimmed mean = ' + str(stats.trim_mean(np.array(recalls), 0.02)))