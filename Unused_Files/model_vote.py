#Unused File
import math
import random
from os import path
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif, f_classif
import pickle
import mpl_toolkits as mpl
import matplotlib as mpl
import matplotlib.ticker
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


x_ticks=[]
k_accuracies=[]
k_recalls=[]

line_number=1

#Lower the threshold from 0.5 to 0.2 in order to retrieve positive results that would otherwise be negative when the model lacks confidence i.e probabilty 0.45
proba_threshold = 0.33
proba_threshold_svm = 0.24 #or 0.2
proba_threshold_lr = 0.5

#load the credit card csv file
credit_data_df = pd.read_csv("../data/dev_data.csv")
# create a dataframe of zeros   |
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]
print(credit_data_df_legit)
# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

print(credit_data_df.shape)
# count ones |
#no. of rows
numberOfOnes = credit_data_df_fraud.shape[0]

load_balancing_ratio = 1.0

numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)
index = ['Ones', 'Zeros']
random_seeds = [12, 23, 34, 1, 56, 67, 45, 6]
#random_seeds = [23]

load_balancing_ratio_lr = 1.0
# **load-balancing**
numberOfZeros_lr = math.floor(load_balancing_ratio_lr * numberOfOnes)

#Method to plot the ROC curve
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

features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
# Array to store the accuracies and the recalls
accuracies = []
recalls = []
precisions = []
f1_scores = []


for rs in random_seeds:
    print('Random Seed value is ' + str(rs))
    # choose a random sample of zeros
    credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

    # merge the above with the ones and do the rest of the pipeline with it
    result = credit_data_df_legit_random.append(credit_data_df_fraud)

    # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 (dtataframe subsetin)
    X = result[features]

    # create array y, which includes the classification only
    y = result['Class']


    #Select the best features
    select_kbest = SelectKBest(mutual_info_classif, k=26)
    #Fit the method onto the data and then return a transformed array
    X_new =select_kbest.fit_transform(X, y)
    mask = select_kbest.get_support()

    select_kbest = SelectKBest(f_classif, k=5)
    X_new_svm = select_kbest.fit_transform(X, y)

    select_kbest = SelectKBest(f_classif, k=24)
    X_new_lr = select_kbest.fit_transform(X, y)

    # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split


    # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_new_svm, y, test_size=0.2, random_state=rs, stratify=y)
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_new_lr, y, test_size=0.2, random_state=rs,
                                                                    stratify=y)

    # use sklearns random forest to fit a model to train data
    clf = RandomForestClassifier(n_estimators=100, random_state=rs,class_weight='balanced')
    clf.fit(X_train, y_train)
    ml_object = [clf, mask]
    #use the model
    #pickle.dump(model_and_features, open(path.join('models', 'rf.pkl'), 'wb'))

    # for this classification use Predict_proba to give the probability of a 1(fraud)
    probs = clf.predict_proba(X_test)
    preds = probs[:, 1]

    y_pred = [1 if x >= proba_threshold else 0 for x in preds]

    clf_svm = svm.SVC(C=1, kernel='linear', probability=True, random_state=rs, class_weight='balanced')
    clf_svm.fit(X_train_svm, y_train_svm)

    # use the model
    # pickle.dump(model_and_features, open(path.join('models', 'svc2.pkl'), 'wb'))
    # y_pred = clf.predict(X_test)
    probs_svm = clf_svm.predict_proba(X_test_svm)
    preds_svm = probs_svm[:, 1]
    # if probability  is above the threshold classify as a 1
    y_pred_svm = [1 if x >= proba_threshold_svm else 0 for x in preds_svm]

    ############################


    # Select the 20 best features
    # f_classif 24


    # use sklearns random forest to fit a model to train data
    clf_lr = LogisticRegression(random_state=rs, solver='liblinear', class_weight='balanced', max_iter=1000)
    clf_lr.fit(X_train_lr, y_train_lr)
    # print(clf.fit(X_train,y_train))


    # for this classification use Predict_proba to give the probability of a 1(fraud)
    probs_lr = clf_lr.predict_proba(X_test_lr)
    preds_lr = probs_lr[:, 1]
    y_pred_lr = [1 if x >= proba_threshold_lr else 0 for x in preds_lr]

    ############################
    y_pred_vote = []
    for i in range(len(y_pred_lr)):
        lr_vote = y_pred_lr[i]
        rf_vote = y_pred[i]
        svm_vote = y_pred_svm[i]
        vote_sum = lr_vote+rf_vote+svm_vote
        vote = 1 if vote_sum>1 else 0
        y_pred_vote.append(vote)


    # use sklearn metrics to judge accuracy of model using test data
    acc = accuracy_score(y_test, y_pred_vote)
    accuracies.append(acc)
    # output score
    print(acc)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    target_names = ['class 0', 'class 1']
    cm = confusion_matrix(y_test,y_pred)
    print('Below is the confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print((tn, fp, fn, tp))

    recall = tp / (tp + fn)
    recalls.append(recall)

    precision = tp / (tp + fp)
    precisions.append(precision)

    f1_score = 2 * ((precision * recall) / (precision + recall))
    f1_scores.append(f1_score)

    print(classification_report(y_test, y_pred, target_names=target_names))

    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    observations_df = pd.DataFrame(columns = ['y_true', 'prediction', 'proba'])
    observations_df['y_true'] = y_test
    observations_df['prediction'] = y_pred
    observations_df['proba'] = preds

#plot_roc()
print(threshold)

plt.show()


#calculate the mean accuracy
mean_accuracy = np.mean(np.array(accuracies))
#Calculate the mean recalldf.plot.scatter(x='Time',y='Number_of_Vehicles', s=1)
mean_recall = np.mean(np.array(recalls))

print('accuracy mean = ' + str(mean_accuracy))
print('recall mean = ' + str(mean_recall))
print('precision mean = ' + str(np.mean(np.array(precisions))))
print('F1 mean = ' + str(np.mean(np.array(f1_scores))))

