#Unused File
import math
import random
from os import path
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
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

x_ticks=[]
k_accuracies=[]
k_recalls=[]

line_number=1

#Lower the threshold from 0.5 to 0.2 in order to retrieve positive results that would otherwise be negative when the model lacks confidence i.e probabilty 0.45
proba_threshold = 0.5

#load the credit card csv file
credit_data_df = pd.read_csv("../data/dev_data.csv")
# create a dataframe of zeros   |
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

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
test_size = [0.1, 0.2, 0.3, 0.4]
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

for ts in test_size:
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

        # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=ts, random_state=rs, stratify=y)


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

        # use sklearn metrics to judge accuracy of model using test data
        acc = accuracy_score(y_test, y_pred)
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
        print(classification_report(y_test, y_pred, target_names=target_names))

        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)

        observations_df = pd.DataFrame(columns = ['y_true', 'prediction', 'proba'])
        observations_df['y_true'] = y_test
        observations_df['prediction'] = y_pred
        observations_df['proba'] = preds



    #calculate the mean accuracy
    mean_accuracy = np.mean(np.array(accuracies))
    #Calculate the mean recall
    mean_recall = np.mean(np.array(recalls))

    print('accuracy mean = ' + str(mean_accuracy))
    print('recall mean = ' + str(mean_recall))
    k_accuracies.append(mean_accuracy)
    k_recalls.append(mean_recall)

fig, ax = plt.subplots()
plt.plot(test_size, k_accuracies)
plt.ylabel('Accuracies')
plt.xlabel('Test-size')
plt.title('RF - Different Test sizes')
ax.set_xticks(np.arange(0.1, 0.4, 0.05))
# plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
plt.show()

fig, ax = plt.subplots()
plt.plot(test_size, k_recalls)
plt.ylabel('Recalls')
plt.xlabel('Test-size')
plt.title('RF - Different Test sizes')
ax.set_xticks(np.arange(0.1, 0.4, 0.05))
# plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
plt.show()

print(np.mean(np.array(k_accuracies)))
#Calculate the mean recall
print(np.mean(np.array(k_recalls)))