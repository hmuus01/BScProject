#SUPPORT VECTOR MACHINE
import math
import pickle
from os import path

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import svm
from scipy import stats

import matplotlib.pyplot as plt

x_ticks=[]
k_accuracies=[]
k_recalls=[]
line_number=1

proba_threshold = 0.4

#Step 1 Load the credit card csv file
credit_data_df = pd.read_csv("data/creditcard.csv")

#Step 2
# create a dataframe of zeros   | example rslt_df = dataframe[dataframe['Percentage'] > 80]
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

#count ones |
# numberOfOnes = credit_data_df_fraud.shape[0]
# load_balancing_ratio = 1.0
# numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)

random_seeds = [12, 23, 34, 1, 56, 67, 45, 6, 9, 10, 11, 12, 13,14,15, 16, 27, 18]
# all_recalls={'lbfgs':[], 'newton-cg':[]}
all_accuracys={'lbfgs':[], 'newton-cg':[], 'liblinear':[]}
all_recalls = {'lbfgs':[], 'newton-cg':[], 'liblinear':[]}
lb_range=range(1,20)

optimizers=['lbfgs','newton-cg','liblinear']

#lbfgs — Stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno.

feature_headers = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
for optimizer in optimizers:
    for load_balancing_ratio in lb_range:
        # Array to store the accuracies and the recalls
        accuracies = []
        recalls = []
        # count ones | No.Rows
        numberOfOnes = credit_data_df_fraud.shape[0]
        # **load-balancing**
        numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)
        #print('number of zeros is: ' + str(numberOfZeros))
        for rs in random_seeds:
            #print('random seed is' + str(rs))
            # choose a random sample of zeros
            credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

            # merge the above with the ones and do the rest of the pipeline with it
            result = credit_data_df_legit_random.append(credit_data_df_fraud)

            # create array X, which includes variables time, amount, V1, V2, V3, V4 (dtataframe subsetin)
            X = result[feature_headers]

            # create array y, which includes the classification only
            y = result['Class']

            # Select the 20 best features
            select_kbest = SelectKBest(f_regression, k=29)
            X_new = select_kbest.fit_transform(X, y)
            mask = select_kbest.get_support()

            # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=y)

            # use sklearns random forest to fit a model to train data
            # print('################################################################################################')
            # print('rs is ' + str(rs) + ' optimizer is ' + str(optimizer) + ' load balancing ratio is ' + str(load_balancing_ratio))
            # print('------------------------------------------------------------------------------------------------')

            clf = LogisticRegression(random_state=rs, solver=optimizer, class_weight='balanced')
            clf.fit(X_train, y_train)
            ml_object = [clf, mask]
            # use the model
            # pickle.dump(ml_object, open(path.join('models', 'rf.pkl'), 'wb'))
            # y_pred = clf.predict(X_test)
            # for this classification use Predict_proba to give the only probability of 1
            probs = clf.predict_proba(X_test)

            preds = probs[:, 1]

            y_pred = [1 if x >= proba_threshold else 0 for x in preds]

            # use sklearn metrics to judge accuracy of model using test data
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            # output score
            #print(acc)

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

            observations_df = pd.DataFrame(columns=['y_true', 'prediction', 'proba'])
            observations_df['y_true'] = y_test
            observations_df['prediction'] = y_pred
            observations_df['proba'] = preds

        #Threshold

        #ROC prob
        # use select k_best from sklearn to choose best features
        mean_accuracy = np.mean(np.array(accuracies))
        #Discards outliers
        mean_recall = stats.trim_mean(np.array(recalls), 0.1)


        #print(mean_recall)
        all_recalls[optimizer].append(mean_recall)
        # print('optimizer is ' + ' -> ' + str(optimizer) + ' lb ratio is ' + ' -> ' + str(
        #     load_balancing_ratio) + '  recalls  are ' + ' -> ' + str(recalls))
        # print('optimizer is ' + ' -> ' + str(optimizer) + ' lb ratio is ' + ' -> ' + str(load_balancing_ratio) + ' All recalls  optimzer is '+ ' -> ' + str(all_recalls[optimizer]))
        all_accuracys[optimizer].append(mean_accuracy)
        # print('optimizer is ' + ' -> ' + str(optimizer) + ' lb ratio is ' + ' -> ' + str(load_balancing_ratio) + ' All acc optimzer is ' + ' -> ' + str(all_accuracys[optimizer]))
        print('accuracy mean = ' + str(mean_accuracy))
        print('recall mean = ' + str(mean_recall))

        print('------------------------------------------------------------------------------------------------')



        #k_accuracies.append(mean_accuracy)
        #k_recalls.append(mean_recall)



# # import matplotlib.pyplot as plt
plt.title('Load-Balancing Test on Recalls')
plt.plot(lb_range, all_recalls['lbfgs'], label='lbfgs')
plt.plot(lb_range, all_recalls['newton-cg'], label='newton-cg')
#plt.plot(range(len(all_recalls['sag'])), all_recalls['sag'], label='sag')
plt.plot(lb_range, all_recalls['liblinear'], label='liblinear')
plt.ylabel('Recalls')
plt.xlabel('Load-Balancing Ratio')
plt.legend()
plt.show()
plt.title('Load-Balancing on Accuracies')
plt.plot(lb_range, all_accuracys['lbfgs'], label='lbfgs')
plt.plot(lb_range, all_accuracys['newton-cg'], label='newton-cg')
#plt.plot(range(len(all_accuracys['sag'])), all_accuracys['sag'], label='sag')
plt.plot(lb_range, all_accuracys['liblinear'], label='liblinear')
plt.ylabel('Accuracies')
plt.xlabel('Load-Balancing Ratio')
plt.legend()
plt.show()
