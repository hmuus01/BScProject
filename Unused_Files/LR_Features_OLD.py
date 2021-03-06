#Unused File
import math
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
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, f_oneway,f_regression ,mutual_info_classif ,mutual_info_regression
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
all_accuracys = {'lbfgs':[], 'newton-cg':[], 'liblinear':[]}
all_recalls = {'lbfgs':[], 'newton-cg':[], 'liblinear':[]}

#load the credit card csv file
credit_data_df = pd.read_csv("../data/dev_data.csv")
test_data_df = pd.read_csv("../data/edited_unused_credit.csv")


# create a dataframe of zeros   |
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]


# count ones |
#no. of rows
numberOfOnes = credit_data_df_fraud.shape[0]
realZeros = credit_data_df_legit.shape[0]
load_balancing_ratio = 1.0
numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)
index = ['Ones', 'Zeros']
random_seeds = [12, 23, 34, 1, 56, 67, 45, 6]

f_range = range(1,30)

# df = pd.DataFrame({'Ones': numberOfOnes, 'Zeros': numberOfZeros}, index=index)

# df = pd.DataFrame({'Values':['Num. Ones', 'Num. Zeros'], 'No.':[numberOfOnes, realZeros]})
# ax = df.plot.bar(x='Values', y='No.', rot=0)
#
# #ax = df.plot.bar(rot=0)
# plt.show()

optimizers = ['lbfgs', 'newton-cg', 'liblinear']

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
for optimizer in optimizers:
    for k in range(1,len(feature_headers)):
        # Array to store the accuracies and the recalls
        accuracies = []
        recalls = []
        for rs in random_seeds:
            # choose a random sample of zeros
            credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

            # merge the above with the ones and do the rest of the pipeline with it
            result = credit_data_df_legit_random.append(credit_data_df_fraud)
            result = result.sample(frac=1, random_state=rs)
            # **load-balancing**

            # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 (dtataframe subsetin)
            X = result[feature_headers]


            # create array y, which includes the classification only
            y = result['Class']

            #Select the 20 best features
            select_kbest = SelectKBest(f_regression, k=k)
            X_new =select_kbest.fit_transform(X, y)
            mask = select_kbest.get_support()

            # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y)

            # use sklearns random forest to fit a model to train data
            clf = LogisticRegression(random_state=rs, solver=optimizer, class_weight='balanced')
            clf.fit(X_train, y_train)
            ml_object = [clf, mask]
            #use the model
            #pickle.dump(model_and_features, open(path.join('models', 'rf.pkl'), 'wb'))
            #y_pred = clf.predict(X_test)

            # for this classification use Predict_proba to give the probability of a 1(fraud)
            probs = clf.predict_proba(X_test)
            # print('THis is PROBS')
            # print(probs)
            # print('#######################')
            preds = probs[:, 1]
            # print('THis is preds')
            # print(preds)
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
        print('k= ' + str(k))
        print('accuracy mean = ' + str(mean_accuracy))
        print('recall mean = ' + str(mean_recall))

        k_accuracies.append(mean_accuracy)
        k_recalls.append(mean_recall)
        x_ticks.append(k)


# plt.plot(x_ticks, k_accuracies)
# plt.ylabel('Accuracies')
# plt.xlabel('Features')
# plt.title('Features Test on Accuracies')
# plt.xticks(x_ticks)
# plt.show()
# 
# plt.plot(x_ticks, k_recalls)
# plt.ylabel('Recalls')
# plt.title('Features Test on Recalls')
# plt.xticks(x_ticks)
# plt.xlabel('Features')
# plt.show()

# # import matplotlib.pyplot as plt
plt.title('Load-Balancing Test on Recalls')
plt.plot(range(all_recalls['lbfgs']), all_recalls['lbfgs'], label='lbfgs')
plt.plot(range(all_recalls['newton-cg']), all_recalls['newton-cg'], label='newton-cg')
#plt.plot(range(len(all_recalls['sag'])), all_recalls['sag'], label='sag')
plt.plot(range(all_recalls['liblinear']), all_recalls['liblinear'], label='liblinear')
plt.ylabel('Recalls')
plt.xlabel('Features Test')
plt.legend()
plt.show()

plt.title('Load-Balancing on Accuracies')
plt.plot(range(all_accuracys['lbfgs']), all_accuracys['lbfgs'], label='lbfgs')
plt.plot(range(all_accuracys['newton-cg']), all_accuracys['newton-cg'], label='newton-cg')
#plt.plot(range(len(all_accuracys['sag'])), all_accuracys['sag'], label='sag')
plt.plot(range(all_accuracys['liblinear']), all_accuracys['liblinear'], label='liblinear')
plt.ylabel('Accuracies')
plt.xlabel('Features Test')
plt.legend()
plt.show()