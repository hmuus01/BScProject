import math
import random
from os import path
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
import pickle

import matplotlib.pyplot as plt

x_ticks=[]
k_accuracies=[]
k_recalls=[]

line_number=1

#Lower the threshold from 0.5 to 0.2 in order to retrieve positive results that would otherwise be negative when the model lacks confidence i.e probabilty 0.45
proba_threshold = 0.5

# #Array to store the accuracies and the recalls
# accuracies= []
# recalls = []

#load the credit card csv file
credit_data_df = pd.read_csv("data/creditcard.csv")
test_data_df = pd.read_csv("data/credit.csv")

# create a dataframe of zeros   |
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

# print('This Legit')
# print(credit_data_df_legit)
# print('#####################')
# print('This Fraud')
# print(credit_data_df_fraud)
# print('--------------------')

# count ones |
#no. of rows
numberOfOnes = credit_data_df_fraud.shape[0]
realZeros = credit_data_df_legit.shape[0]
load_balancing_ratio = 1.0
# **load-balancing**
numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)
index = ['Ones', 'Zeros']
#random_seeds = [12, 23, 34, 1, 56, 67, 45, 6]
random_seeds = set(random.sample(range(1, 100), 20))
# df = pd.DataFrame({'Ones': numberOfOnes, 'Zeros': numberOfZeros}, index=index)

# df = pd.DataFrame({'Values':['Num. Ones', 'Num. Zeros'], 'No.':[numberOfOnes, realZeros]})
# ax = df.plot.bar(x='Values', y='No.', rot=0)
#
# #ax = df.plot.bar(rot=0)
# plt.show()

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

feature_headers = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
# Array to store the accuracies and the recalls
accuracies = []
recalls = []
for rs in random_seeds:
    # choose a random sample of zeros
    credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

    # merge the above with the ones and do the rest of the pipeline with it
    result = credit_data_df_legit_random.append(credit_data_df_fraud)


    # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 (dtataframe subsetin)
    X = result[feature_headers]

    # create array y, which includes the classification only
    y = result['Class']

    #Select the 20 best features
    select_kbest = SelectKBest(f_regression, k=29)
    X_new =select_kbest.fit_transform(X, y)
    mask = select_kbest.get_support()

    # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y)

    # use sklearns random forest to fit a model to train data
    clf = RandomForestClassifier(n_estimators=100, random_state=rs,class_weight='balanced')
    clf.fit(X_train, y_train)
    ml_object = [clf, mask]
    #use the model
    #pickle.dump(ml_object, open(path.join('models', 'rf.pkl'), 'wb'))
    #y_pred = clf.predict(X_test)

    # for this classification use Predict_proba to give the probability of a 1(fraud)
    probs = clf.predict_proba(X_test)
    # print('THis is PROBS')
    #print(probs)
    # print('#######################')
    preds = probs[:, 1]
    # print('THis is preds')
    #print(preds)
    # print('---------------------')

    y_pred = [1 if x >= proba_threshold else 0 for x in preds]
    # print('THis is Y_preds')
    # print(y_pred)
    # print('NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN')

    # use sklearn metrics to judge accuracy of model using test data
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    # output score
    print(acc)


    # probs1 = clf.predict_proba(X_test1)
    # preds1 = probs1[:, 1]
    # print('==================')
    # print(probs1)
    # print(preds1)
    # print('==================')
    # precision / recall
    # confusion matrix |
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    target_names = ['class 0', 'class 1']
    cm = confusion_matrix(y_test,y_pred)
    print(confusion_matrix(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print((tn, fp, fn, tp))

    # import scikitplot as skplt
    #
    # skplt.metrics.plot_confusion_matrix(y_test, y_pred)

    recall = tp / (tp + fn)
    recalls.append(recall)
    print(classification_report(y_test, y_pred, target_names=target_names))

    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    observations_df = pd.DataFrame(columns = ['y_true', 'prediction', 'proba'])
    observations_df['y_true'] = y_test
    observations_df['prediction'] = y_pred
    observations_df['proba'] = preds
print(observations_df.shape)
print(y_train.shape)
print(X_train.shape)
print(result.shape)

# cm1 = pd.DataFrame(cm)
# cm1.index.name = 'Actual'
# cm1.columns.name = 'Predicted'
# #plt.figure(figsize = (10,10))
# annot_kws = {"ha": 'center',"va": 'center'}
# sns.heatmap(cm1,cmap= "Blues",annot=True, annot_kws=annot_kws)
# plt.show()

    # method I: plt
# plt.scatter(y_test, y_2)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

#plot_roc()
#Threshold
#ROC prob

#calculate the mean accuracy
mean_accuracy = np.mean(np.array(accuracies))
#Calculate the mean recall
mean_recall = np.mean(np.array(recalls))

print('accuracy mean = ' + str(mean_accuracy))
print('recall mean = ' + str(mean_recall))

