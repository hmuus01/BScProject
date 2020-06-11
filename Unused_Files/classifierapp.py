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

# probability threshold
proba_threshold = 0.5

# #Array to store the accuracies and the recalls
# accuracies= []
# recalls = []

#load the credit card csv file
credit_data_df = pd.read_csv("../data/dev_data.csv")
test_data_df = pd.read_csv("../data/edited_unused_credit.csv")
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
# print(numberOfOnes)
realZeros = credit_data_df_legit.shape[0]
# print(realZeros)
load_balancing_ratio = 1.0
# **load-balancing**
numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)
index = ['Ones', 'Zeros']

random_seeds = [12, 23, 34, 1, 56, 67, 45, 6]



#random_seeds = set(random.sample(range(1, 100), 20))
# df = pd.DataFrame({'1': numberOfOnes, '0': numberOfZeros}, index=index)

# df = pd.DataFrame({'Values':['1', '0'], 'No.':[numberOfOnes, numberOfZeros]})
# ax = df.plot.bar(x='Values', y='No.', rot=0, legend=None)
#
# #ax = df.plot.bar(rot=0)
# plt.title('Credit Card Fraud Dataset Class Frequency')
# plt.xlabel('Class')
# plt.ylabel('Frequency')
# plt.show()

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
#features = random.sample(features, len(features))
accuracies = []
recalls = []

#ime = credit_data_df[features[0]]
#print(credit_data_df['Time'])
#credit_data_df['Time'] = credit_data_df['Time'].astype('int64')
#ax = credit_data_df['Time'].plot.hist()
#ax.set_title("Distribution of Time")
#ax.set_xlabel('Time')
# ax.yaxis.set_major_formatter(
# mpl.ticker.EngFormatter(places=0))
cr = credit_data_df.head(10)
print(cr)

# #time = credit_data_df_fraud['Amount'].values
# #time = credit_data_df_legit.sample(numberOfOnes)['Amount'].values
# time = credit_data_df['Time'].values
#
# fig, ax = plt.subplots()
# sns.distplot(time, ax=ax, color='r', bins=24)
# ax.set_title('Distribution of Transaction Time', fontsize=14)
# ax.set_xlim([min(time), max(time)])
#sns.pairplot(credit_data_df, palette ='coolwarm')

#plt.savefig("AgeofDrivers.png", dpi=300, bbox_inches='tight')
plt.show()

for rs in random_seeds:
    print('Random Seed value is ' + str(rs))
    # choose a random sample of zeros
    credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

    # print('Credit data legit')
    # print(credit_data_df_legit)
    # print('Credit data legit random')
    # print(credit_data_df_legit_random)

    # merge the above with the ones and do the rest of the pipeline with it
    result = credit_data_df_legit_random.append(credit_data_df_fraud)


    # print('Result is ')
    # print(result)
    #Shuffle the result
    #result = result.sample(frac=1, random_state=rs)

    # print('Result NEW is ')
    # print(result)

    # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 (dtataframe subsetin)
    X = result[features]

    # create array y, which includes the classification only
    y = result['Class']

    # print('X')
    # print(X)
    #Select the 20 best features
    select_kbest = SelectKBest(mutual_info_classif, k=29)
    X_new =select_kbest.fit_transform(X, y)
    mask = select_kbest.get_support()

    # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y)

    # print('X_new')
    # print(X_new)
    # print('y')
    # print(y)
    # use sklearns random forest to fit a model to train data
    clf = RandomForestClassifier(n_estimators=100, random_state=rs,class_weight='balanced')
    clf.fit(X_train, y_train)
    # print(clf.fit(X_train,y_train))
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



# print("random credit data " + str(credit_data_df_legit_random.shape[0]))
# print("result " + str(result.shape[0]))
# print(X_test.shape)
# print(y_test.shape)
#plot_roc()
#Plot Confusion Matrix
# ax = plt.subplot()
# sns.heatmap(cm, ax=ax, annot=True, cmap=plt.cm.Reds)
# ax.set_title("Random Forest \n Confusion Matrix", fontsize=14)
# ax.set_xlabel("Predicted Label")
# ax.set_ylabel("Actual Label")
#plt.show()

# print(observations_df.shape)
# print(y_train.shape)
# print(X_train.shape)
# print(result.shape)
# print(X_test.shape)
# print(y_test.shape)
# print(len(y_pred))

# plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.show()


#Threshold
#ROC prob

#calculate the mean accuracy
mean_accuracy = np.mean(np.array(accuracies))
#Calculate the mean recall
mean_recall = np.mean(np.array(recalls))

print('accuracy mean = ' + str(mean_accuracy))
print('recall mean = ' + str(mean_recall))

# cv_range = 10 #[1,2,3,4,5,6,7,8,9,10]
# cross_v = cross_val_score(clf, X_train, y_train, cv=cv_range,)

#print('Cross validation is ' + str(cross_v))

# print('Cross_v mean is ' + str(cross_v.mean()))
#cm1 = pd.DataFrame(cross_v)
# plt.plot(cross_v)
# plt.ylabel('Cross Validation Score')
# plt.xlabel('Cross Validation')
# plt.title('Cross_val_score plot')
# #plt.xticks(cv_range)
# plt.show()

#ax = cm1.plot.hist(legend=None)
#ax = sns.kdeplot(int(cross_v.mean()))
# ax = sns.distplot(cm1, hist=True, kde=True)
# ax.set_title('-')
# ax.set_xlabel('Cross Validation')
# plt.show()

