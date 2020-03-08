#SUPPORT VECTOR MACHINE
import math
import random

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

import matplotlib.pyplot as plt


proba_threshold = 0.5

x_ticks=[]
accuracies= []
recalls = []
credit_data_df = pd.read_csv("data/creditcard.csv")

# create a dataframe of zeros   | example rslt_df = dataframe[dataframe['Percentage'] > 80]
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

# count ones |
numberOfOnes = credit_data_df_fraud.shape[0]
load_balancing_ratio = 1.0
#numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)
#num_randoms = 10
#random_seeds = set(random.sample(range(1, 100), num_randoms)) #[12, 23, 34, 1, 56]#, 67, 45, 6]
#print(random_seeds)

all_accuracys = {'lbfgs': [], 'newton-cg': []}
lb_range=range(1, 30)
optimizers=['lbfgs', 'newton-cg']

for load_balancing_ratio in lb_range:
    for optimizer in optimizers:
        numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)
        num_randoms = 10

        random_seeds = set(random.sample(range(1, 100), 10))


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


        for rs in random_seeds:
            print(rs)
            # choose a random sample of zeros
            credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)
            credit_data_df_legit_random = credit_data_df_legit_random.sample(frac=1, random_state=rs).reset_index(
                drop=True)

            # shufle both dataframes
            credit_data_df_fraud = credit_data_df_fraud.sample(frac=1, random_state=rs).reset_index(drop=True)

            # generate test set with 50 legitimate and 50 fraudulent transactions:
            df1 = credit_data_df_legit_random.iloc[:50, :]
            df2 = credit_data_df_fraud.iloc[:50, :]
            #
            df3 = credit_data_df_legit_random.iloc[50:, :]
            df4 = credit_data_df_fraud.iloc[50:, :]
            # Test dataframe
            test_df = df1.append(df2)
            # merge the above with the ones and do the rest of the pipeline with it
            result = df3.append(df4)
            # Test features
            X_test = test_df[
                ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
                 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
            # Test class(labels)
            y_test = test_df['Class']

            # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 (dtataframe subsetin)
            X_train = result[
                ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
                 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]

            # create array y, which includes the classification only
            y_train = result['Class']

            # use sklearns random forrest to fit a model to train data
            # clf = svm.SVC(gamma='scale', probability=True, kernel='linear') class_weight={1: 5}

            # clf = svm.SVC(C=1, kernel='linear', probability=True, random_state=0, class_weight={1: 2})
            clf = LogisticRegression(random_state=0, solver=optimizer, class_weight={1: int(load_balancing_ratio)})

            clf.fit(X_train, y_train)
            # y_pred = clf.predict(X_test)
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

            observations_df = pd.DataFrame(columns=['y_true', 'prediction', 'proba'])
            observations_df['y_true'] = y_test
            observations_df['prediction'] = y_pred
            observations_df['proba'] = preds
            # method I: plt
            # plot_roc()

        # Threshold
        # ROC prob
        # use select k_best from sklearn to choose best features
        mean_accuracy = np.mean(np.array(accuracies))
        mean_recall = np.mean(np.array(recalls))
        all_accuracys[optimizer].append(mean_accuracy)
        print('accuracy mean = ' + str(mean_accuracy))
        print('recall mean = ' + str(mean_recall))



import matplotlib.pyplot as plt
#plt.plot(lb_range, all_recalls['lbfgs'], label='lbfgs')
#plt.plot(lb_range, all_recalls['newton-cg'], label='newton-cg')
plt.plot(lb_range, all_accuracys['lbfgs'], label='lbfgs')
plt.plot(lb_range, all_accuracys['newton-cg'], label='newton-cg')
#plt.ylabel('recalls')
plt.legend()
plt.ylabel('Accuracies %')
plt.xlabel('LB ratio')
plt.title("Load Balancing Ratio's  ")
plt.show()


#Histogram & boxplot of accuracies and recalls

#Tod o: Visualize observations (zeros and ones)
## Play with sample weight
## try with different load balancing levels like 1:2 or 1:4 etc.
# Plot probability distributions
# Plot the hyperplanes

