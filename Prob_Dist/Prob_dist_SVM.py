#In this file the probability distribtuion graph for SVM is plotted for 8 random seeds.
#Import Statements
import math
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import svm
import matplotlib.pyplot as plt


# Function to plot the distribution of the probability predictions, this takes as parameters the name of the model|
#the prob prediction values of  the SVM model and the title of the plot
def generate_density_plot(model_name, true_positives, true_negatives):
    if len(true_negatives) < 1 or len(true_positives) < 1:
        return

    def plot_pdf():
        #Use Seaborn distribution plot function to plot the distribution of the true positivevalue(fraud) predictions For SVM
        sns.distplot(true_positives, hist=False, kde=True, kde_kws={'linewidth': 1}, label='TP')
        #Use Seaborn distribution plot function to plot the distribution of the true negative value(Legit) predictions For SVM
        sns.distplot(true_negatives, hist=False, kde=True, kde_kws={'linewidth': 1}, label='TN')
        # Plot formatting
        plt.legend(prop={'size': 10}, title='Prediction Distribution')
        #Title of the plot
        plt.title('Density Plot for Probability Prediction\n' + model_name)
        #Label the x-axis
        plt.xlabel('Probability Prediction is 1')
        #Label the y-axis
        plt.ylabel('Density')
        # Save the plot | Uncomment to save the plot
        #plt.savefig('RF' + '_prob_dist.png', bbox_inches='tight')
        #Show the plot
        plt.show()

    #Set the range of the X-axis labels
    plt.xlim([0.0, 1.0])
    # Set the range of the Y-axis labels
    plt.ylim([0, 7])
    #call to the plot pdf method if there are still values in the svm array
    plot_pdf()

#Probability threshold to classify a transaction
proba_threshold = 0.5

#load the credit card csv file
credit_data_df = pd.read_csv("../data/dev_data.csv")
# create a dataframe of zeros   |
credit_data_df_legit = credit_data_df[credit_data_df['Class'] == 0]

# create a dataframe of 1s only |
credit_data_df_fraud = credit_data_df[credit_data_df['Class'] == 1]

# count ones |
#no. of rows
numberOfOnes = credit_data_df_fraud.shape[0]

#LBR set to 1 | After testing this was found to be the best LB ratio for Support Vector Machines
load_balancing_ratio = 1.0

#Set the number of zero's variable to be equal to the number of ones
numberOfZeros = math.floor(load_balancing_ratio * numberOfOnes)

#A list(8) of random seed's used for the random state parameter as way to counteract overfitting and get the mean
#of how the model performs from different data and different train-test splits
random_seeds = [12, 23, 34, 1, 56, 67, 45, 6]

#List of features used in training the model
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Array to store the accuracy, recalls and true positive/negative probability predictions
accuracies = []
recalls = []
tp_probs=[]
tn_probs=[]

#Train & Test the model using different random seeds
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

    # Select the best features | After Testing this was found to be the best amount of features for Support Vector Machines
    select_kbest = SelectKBest(f_classif, k=5)
    # Fit the method onto the data and then return a transformed array
    X_new = select_kbest.fit_transform(X, y)

    # use sklearn to split the X and y, into X_train, X_test, y_train y_test with 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=rs, stratify=y) #,kernel='poly', degree=2,

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    #                                                    TRAINING ON THE TRAINING SET
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # use sklearns Support Vector Machines to fit a model to train data
    clf = svm.SVC(C=1, kernel='linear', probability=True, random_state=rs, class_weight='balanced')

    #Train the model using the training data, meaning learn about the relationship between feature and output class
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    #                                                           TESTING ON THE TEST SET
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    # for this classification use Predict_proba to give the probability of the classes whereas predict() just predicts the output class for the test set
    probs = clf.predict_proba(X_test)

    # store just the fraudulent class probabilities
    fraudulent_class_probabilities = probs[:, 1]

    # Classify whether a transaction is legit or fraud depending on if it above or below the threshold value
    y_pred = [1 if x >= proba_threshold else 0 for x in fraudulent_class_probabilities]

    # Loop through the fraudulent probabilities
    for i in range(len(fraudulent_class_probabilities)):
        # if the probability of the transaction in the test set is fraud add it to the list of true positive probability predictions
        if y_test.values[i] == 1:
            tp_probs.append(fraudulent_class_probabilities[i])
        # if the probability of the transaction is legit add it to the list of true negative probability predictions
        else:
            tn_probs.append(fraudulent_class_probabilities[i])

    # use sklearn metrics to judge accuracy of model using test data
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Print the output score
    print(acc)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    # Legit - Class 0, Fraud - Class 1
    target_names = ['class 0', 'class 1']

    # Make the confusion matric using the predicted results and check whether its similar to the actual results
    cm = confusion_matrix(y_test, y_pred)

    # Print the confusion matrix
    print('Below is the confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print((tn, fp, fn, tp))

    # Calculate the recall using the formula and add it to the recalls array
    recall = tp / (tp + fn)
    recalls.append(recall)

    # Print the classification report
    print(classification_report(y_test, y_pred, target_names=target_names))


#calculate the mean accuracy & mean recall
mean_accuracy = np.mean(np.array(accuracies))
mean_recall = np.mean(np.array(recalls))

#Print the average accuracy and recall results
print('accuracy mean = ' + str(mean_accuracy))
print('recall mean = ' + str(mean_recall))


#Call the above function with the Support Vector Machines fraud and legit probability predictions as a paramter as well as title of the plot
generate_density_plot('SVM', tp_probs, tn_probs)

