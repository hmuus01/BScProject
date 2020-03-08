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

x_ticks=[]
k_accuracies=[]
k_recalls=[]

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
numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)
#num_randoms = 10
random_seeds = set(random.sample(range(1, 100), 10)) #[12, 23, 34, 1, 56]#, 67, 45, 6]
#print(random_seeds)

#all_accuracys = {'lbfgs': [], 'newton-cg': []}
lb_range=range(1, 30)
feature_headers = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

for load_balancing_ratio in lb_range:
    for rs in random_seeds:
        # choose a random sample of zeros
        credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

        # merge the above with the ones and do the rest of the pipeline with it
        result = credit_data_df_legit_random.append(credit_data_df_fraud)

        # **load-balancing**

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
        clf = RandomForestClassifier(n_estimators=100, random_state=rs, class_weight={1: int(load_balancing_ratio)})
        clf.fit(X_train, y_train)
        ml_object = [clf, mask]
        #use the model
        #pickle.dump(ml_object, open(path.join('models', 'rf.pkl'), 'wb'))
        #y_pred = clf.predict(X_test)
        # for this classification use Predict_proba to give the only probability of 1
        probs = clf.predict_proba(X_test)
        preds = probs[:, 1]

        y_pred = [1 if x >= proba_threshold else 0 for x in preds]

        # use sklearn metrics to judge accuracy of model using test data
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        # output score
        print(acc)

        probs1 = clf.predict_proba(X_test)
        preds1 = probs1[:, 1]
        print('==================')
        print(probs1)
        print(preds1)
        print('==================')
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
        # plot_roc()
    #Threshold
    #ROC prob
    # select k_best from sklearn for best features
    #calculate the mean accuracy
    mean_accuracy = np.mean(np.array(accuracies))
    #Calculate the mean recall
    mean_recall = np.mean(np.array(recalls))
    print('accuracy mean = ' + str(mean_accuracy))
    print('recall mean = ' + str(mean_recall))
    k_accuracies.append(mean_accuracy)
    k_recalls.append(mean_recall)




import matplotlib.pyplot as plt
#plt.plot(lb_range, all_recalls['lbfgs'], label='lbfgs')
#plt.plot(lb_range, all_recalls['newton-cg'], label='newton-cg')
#plt.plot(lb_range, all_accuracys['lbfgs'], label='lbfgs')
plt.plot(lb_range, k_recalls)

#plt.ylabel('recalls')
plt.legend()
plt.ylabel('Recalls %')
plt.xlabel('LB ratio')
plt.title("Load Balancing Ratio's  ")
plt.show()


#Histogram & boxplot of accuracies and recalls

#Tod o: Visualize observations (zeros and ones)
## Play with sample weight
## try with different load balancing levels like 1:2 or 1:4 etc.
# Plot probability distributions
# Plot the hyperplanes

