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
#proba_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
proba_threshold = np.arange(0.1, 1.0, 0.05)
#load the credit card csv file
credit_data_df = pd.read_csv("../data/dev_data.csv")
# create a dataframe of zeros   |
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

# print(credit_data_df.shape)
# count ones |
#no. of rows
numberOfOnes = credit_data_df_fraud.shape[0]
# print(numberOfOnes)
load_balancing_ratio = 1.0
# **load-balancing**
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
recals_pt=[]
prec_pt = []
f1_pt = []
acc_pt = []
for pt in proba_threshold:
    accuracies = []
    recalls = []
    precisions = []
    f1_scores = []
    for rs in random_seeds:
        # print('Random Seed value is ' + str(rs))
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
        #Fit the method onto the data and then returns a transformed array
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


        # for this classification use Predict_proba to give the probability of a 1(fraud)
        probs = clf.predict_proba(X_test)
        preds = probs[:, 1]
        y_pred = [1 if x >= pt else 0 for x in preds]

        # use sklearn metrics to judge accuracy of model using test data
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        # output score
        # print(acc)
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
        target_names = ['class 0', 'class 1']
        cm = confusion_matrix(y_test,y_pred)
        # print('Below is the confusion matrix')
        print(confusion_matrix(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        # print((tn, fp, fn, tp))

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_score=2*((precision*recall)/(precision+recall))
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1_score)
        print("f1 score "+str(f1_score))
        print(f1_scores)
        # print(classification_report(y_test, y_pred, target_names=target_names))

        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)

        observations_df = pd.DataFrame(columns = ['y_true', 'prediction', 'proba'])
        observations_df['y_true'] = y_test
        observations_df['prediction'] = y_pred
        observations_df['proba'] = preds
    print("pt = " + str(pt))
    acc_pt.append(np.mean(np.array(accuracies)))
    recals_pt.append(np.mean(np.array(recalls)))
    prec_pt.append(np.mean(np.array(precisions)))
   #print("f1 scores" + str(f1_scores))
    f1_pt.append(np.mean(np.array(f1_scores)))
    print("f1 mean" + str(np.mean(np.array(f1_scores))))
    print("f1 pt array " + str(f1_pt))
    #Plot Confusion Matrix
    # ax = plt.subplot()
    # sns.heatmap(cm, ax=ax, annot=True, cmap=plt.cm.Reds)
    # ax.set_title("Random Forest \n Confusion Matrix", fontsize=14)
    # ax.set_xlabel("Predicted Label")
    # ax.set_ylabel("Actual Label")
    # plt.show()
    # plot_roc()
    #print(credit_data_df['Amount'].describe())
fig, ax = plt.subplots()
#sns.barplot(x=proba_threshold, y=f1_pt, ax=ax, color='b')
# plt.plot(proba_threshold, recals_pt)
# plt.plot(proba_threshold, prec_pt)
plt.title('RF - Threshold Tests')
#plt.plot(proba_threshold, acc_pt, 'orange', label='Accuracy')
plt.plot(proba_threshold, acc_pt, 'y', label='Acc')
plt.plot(proba_threshold, recals_pt, 'r', label='Recall')
plt.plot(proba_threshold, prec_pt, 'g',label='Precision')
plt.plot(proba_threshold, f1_pt, 'b',label='F1-Score')
ax.set_xticks(np.arange(0.1, 1.0, 0.05))
plt.ylabel('Scores')
plt.xlabel('Thresholds')
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.grid()
plt.show()
#calculate the mean accuracy
mean_prec = np.mean(np.array(prec_pt))
#Calculate the mean recall
mean_recall = np.mean(np.array(recalls))

# print("f1 is" +str(f1_scores))
mean_f1 = np.mean(np.array(f1_scores))
# print('accuracy mean = ' + str(mean_accuracy))
# print('recall mean = ' + str(mean_recall))
print('prec mean = ' + str(mean_prec))


