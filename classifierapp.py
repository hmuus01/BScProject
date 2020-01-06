import math
from os import path
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
import pickle

import matplotlib.pyplot as plt


proba_threshold = 0.2

#Array to store the accuracies and the recalls
accuracies= []
recalls = []
#load the credit card csv file
credit_data_df = pd.read_csv("data/creditcard.csv")

# create a dataframe of zeros   | example rslt_df = dataframe[dataframe['Percentage'] > 80]
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

# count ones |
numberOfOnes = credit_data_df_fraud.shape[0]
load_balancing_ratio = 1.0
numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)

random_seeds = [12, 23, 34, 1, 56, 67, 45, 6]

#Method to plot the ROC curve
def plot_roc():
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

#
for rs in random_seeds:

    # choose a random sample of zeros
    credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

    # merge the above with the ones and do the rest of the pipeline with it
    result = credit_data_df_legit_random.append(credit_data_df_fraud)

    # **load-balancing**

    # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 (dtataframe subsetin)
    X = result[['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']]

    # create array y, which includes the classification only
    y = result['Class']

    X_new = SelectKBest(f_regression, k=20).fit_transform(X, y)

    # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y)


    # use sklearns random forrest to fit a model to train data
    clf = RandomForestClassifier(n_estimators=100, random_state=rs)
    clf.fit(X_train, y_train)

    #use the model
    pickle.dump(clf, open(path.join('models', 'rf.pkl'), 'wb'))
    #y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    preds = probs[:, 1]

    y_pred = [1 if x >= proba_threshold else 0 for x in preds]

    # use sklearn metrics to judge accuracy of model using test data
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    # output score

    print(acc)

    # precision / recall
    # confusion matrix |
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    target_names = ['class 0', 'class 1']
    print(confusion_matrix(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print((tn, fp, fn, tp))

    recall = tp / (tp + fn)
    recalls.append(recall)
    print(classification_report(y_test, y_pred, target_names=target_names))

    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    observations_df = pd.DataFrame(columns = ['y_true', 'prediction', 'proba'])
    observations_df['y_true'] = y_test
    observations_df['prediction'] = y_pred
    observations_df['proba'] = preds
    # method I: plt
    plot_roc()
############################## Logistic REGRESSION (LN)
#Threshold
#ROC prob
# use select k_best from sklearn to choose best features
mean_accuracy = np.mean(np.array(accuracies))
mean_recall = np.mean(np.array(recalls))
print('accuracy mean = ' + str(mean_accuracy))
print('recall mean = ' + str(mean_recall))

