import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

credit_data_df = pd.read_csv("data/creditcard.csv")

# create a dataframe of zeros   | example rslt_df = dataframe[dataframe['Percentage'] > 80]
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

# count ones |
numberOfOnes = credit_data_df_fraud.shape[0]
load_balancing_ratio = 1
numberOfZeros = load_balancing_ratio * numberOfOnes

# choose a random sample of zeros
credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=23)

# merge the above with the ones and do the rest of the pipeline with it
result = credit_data_df_legit_random.append(credit_data_df_fraud)

# **load-balancing**

# create dataframe X, which includes variables time, amount, V1, V2, V3, V4 (dtataframe subsetin)
X = result[['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']]

# create array y, which includes the classification only
y = result['Class']

# use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


# use sklearns random forrest to fit a model to train data
clf = RandomForestClassifier(n_estimators=100, random_state=12)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# use sklearn metrics to judge accuracy of model using test data
acc = accuracy_score(y_test, y_pred)
# output score

print(acc)

# precision / recall
# confusion matrix |
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
target_names = ['class 0', 'class 1']
print(confusion_matrix(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print((tn, fp, fn, tp))
print(classification_report(y_test, y_pred, target_names=target_names))


probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

observations_df = pd.DataFrame(columns = ['y_true', 'prediction', 'proba'])
observations_df['y_true'] = y_test
observations_df['prediction'] = y_pred
observations_df['proba'] = preds
# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Threshold
#ROC prob
# use select k_best from sklearn to choose best features

