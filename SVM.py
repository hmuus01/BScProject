import math
import random
from os import path
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt


#Probability threshold to classify a transaction
proba_threshold = 0.24

#load the credit card csv file
credit_data_df = pd.read_csv("data/dev_data.csv")
# create a dataframe of zeros   |
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

# count ones |
#no. of rows
numberOfOnes = credit_data_df_fraud.shape[0]

#LBR set to 1 in order to have an equal amount of 0's and 1's
load_balancing_ratio = 1.0

#Set the number of zero's variable to be equal to the number of ones
numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)

#A list of random seed's used for the random state parameter as way to counteract overfitting and get the mean
#of how the model performs from different data and different train-test splits
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

#List of features used in training the model
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Array to store the accuracies, recalls, precision and f1_score results
accuracies = []
recalls = []
precisions = []
f1_scores = []

#Train the model using different random seeds
#Do the steps below for each random seed
for rs in random_seeds:
    # choose a random sample of zeros (Legit Class)
    credit_data_df_legit_random = credit_data_df_legit.sample(numberOfZeros, random_state=rs)

    # merge the above with the ones (Fraud Class) and do the rest of the pipeline with it
    result = credit_data_df_legit_random.append(credit_data_df_fraud)

    # create dataframe X, which includes variables time, amount, V1, V2, V3, V4 etc
    X = result[features]

    # create array y, which includes the classification only
    y = result['Class']

    #Select the best features
    select_kbest = SelectKBest(f_classif, k=5)
    #Fit the method onto the data and then return a transformed array
    X_new =select_kbest.fit_transform(X, y)
    #Store the features selected
    mask = select_kbest.get_support()

    # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y)

    # use sklearns Support Vector Machines to fit a model to train data
    clf = svm.SVC(C=1, kernel='linear', probability=True, random_state=rs, class_weight='balanced')

    #Train the model using the training data, meaning learn about the relationship between feature and output class
    clf.fit(X_train, y_train)

   #Save the trained model together with the features selected
    ml_object = [clf, mask]

    #save the model for future use | Uncomment the line below if you would like to save the model
    #pickle.dump(ml_object, open(path.join('models', 'rf.pkl'), 'wb'))

    # for this classification use Predict_proba to give the probability of the classes whereas predict() just predicts the output class for the test set
    probs = clf.predict_proba(X_test)

    #store just the fraudulent class probabilities
    fraudulent_class_probabilities = probs[:, 1]

    #Classify whether a transaction is legit or fraud depending on if it above or below the threshold value
    y_pred = [1 if x >= proba_threshold else 0 for x in fraudulent_class_probabilities]

    # use sklearn metrics to judge accuracy of model using test data
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    #Print the output score
    print(acc)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    target_names = ['class 0', 'class 1']

    #Make the confusion matric using the predicted results and check whether its similar to the actual results
    cm = confusion_matrix(y_test,y_pred)

    #Print the confusion matrix
    print('Below is the confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print((tn, fp, fn, tp))

    #Calculate the recall using the formula and add it to the recalls array
    recall = tp / (tp + fn)
    recalls.append(recall)

    #Calculate the precision using the formula and add it to the precisions array
    precision = tp / (tp + fp)
    precisions.append(precision)

    #Calculate the f1_score using the formula and add it to the f1 scores array
    f1_score = 2 * ((precision * recall) / (precision + recall))
    f1_scores.append(f1_score)

    #Print the classification report
    print(classification_report(y_test, y_pred, target_names=target_names))

    #Plot the points for the Roc curve and the auc score
    fpr, tpr, threshold = metrics.roc_curve(y_test, fraudulent_class_probabilities)
    roc_auc = metrics.auc(fpr, tpr)

    #Store the
    observations_df = pd.DataFrame(columns = ['y_true', 'prediction', 'proba'])
    observations_df['y_true'] = y_test
    observations_df['prediction'] = y_pred
    observations_df['proba'] = fraudulent_class_probabilities

#Display the Roc
#plot_roc()


#calculate the mean accuracy and recall from the respective arrays
mean_accuracy = np.mean(np.array(accuracies))
mean_recall = np.mean(np.array(recalls))

#Print the mean score for all metrics
print('accuracy mean = ' + str(mean_accuracy))
print('recall mean = ' + str(mean_recall))
print('precision mean = ' + str(np.mean(np.array(precisions))))
print('F1 mean = ' + str(np.mean(np.array(f1_scores))))

