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
from sklearn.preprocessing import StandardScaler

x_ticks=[]
k_accuracies=[]
k_recalls=[]

line_number=1

#Lower the threshold from 0.5 to 0.2 in order to retrieve positive results that would otherwise be negative when the model lacks confidence i.e probabilty 0.45
proba_threshold = 0.33

#load the credit card csv file
credit_data_df = pd.read_csv("data/dev_data.csv")
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

    # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y)
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # use sklearns random forest to fit a model to train data
    clf = RandomForestClassifier(n_estimators=100, random_state=rs,class_weight='balanced')
    clf.fit(X_train, y_train)
    ml_object = [clf, mask]
    #use the model
    #pickle.dump(ml_object, open(path.join('models', 'rf.pkl'), 'wb'))

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

# g = sns.catplot('Class', 'Time', data=result)
# g.ax.set_title('Social media preferences by age')
# g.ax.set_ylabel('Social media')
# g.ax.yaxis.set_major_formatter(
# mpl.ticker.EngFormatter(places=0))
# g.ax.set_xlabel('Age')
# plt.show()

# sns.scatterplot('Amount', 'Time', data=result)
# result.plot.scatter(x='Time',y='Amount', s=1)

df = credit_data_df[['Time','Class']]

# ax = credit_data_df_fraud['Time'].plot.hist()
# ax.xaxis.set_major_formatter(
# mpl.ticker.EngFormatter(places=0))
# plt.show()


#bins = np.arange(df['Amount'].min(), df['Amount'].max() + 1, 10)
# ax = df['Amount'].plot.hist(bins = 4)
# ax.set_title('Histogram of the heights of people in the class')
# ax.set_xlabel('Amount')
# ax.set_ylabel('Frequency')
# mean = df.Amount.mean()
# ax.xaxis.set_major_formatter(
# mpl.ticker.EngFormatter(places=0))
# ax.axvline(mean, color='black', linestyle='dashed', linewidth=1)
# ax.annotate('mean={:0.1f}'.format(mean), xy=(mean + 0.2, 5.5), xytext=(mean + 15, 106.2), arrowprops=dict(facecolor='black', shrink=0.5))
# plt.show()
#ax = df['Amount'].plot.box()
from scipy.stats import norm
# #time = credit_data_df['Time'].values

#
# time = credit_data_df_fraud['Time'].values
# fig, ax = plt.subplots()
# sns.distplot(time, ax=ax, color = 'r', bins=24)
# ax.set_title('Distribution of Transaction Time')
# ax.xaxis.set_major_formatter(
# mpl.ticker.EngFormatter(places=0))
# ax.set_xlim([min(time), max(time)])


# ax = df.boxplot(by='Class')
# ax.set_title('-')
# ax.figure.suptitle('')
# ax.set_xlabel('Class')
# ax.yaxis.set_major_formatter(
# mpl.ticker.EngFormatter(places=0))
# ax.set_ylabel('Time or Amount')

plt.show()


#calculate the mean accuracy
mean_accuracy = np.mean(np.array(accuracies))
#Calculate the mean recalldf.plot.scatter(x='Time',y='Number_of_Vehicles', s=1)
mean_recall = np.mean(np.array(recalls))

print('accuracy mean = ' + str(mean_accuracy))
print('recall mean = ' + str(mean_recall))
print('precision mean = ' + str(np.mean(np.array(precisions))))
print('F1 mean = ' + str(np.mean(np.array(f1_scores))))

