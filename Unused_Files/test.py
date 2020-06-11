#Unused File
import pickle
from os import path
import csv
import numpy as np

#client
user_response = 5

with open('../data/edited_unused_credit.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    transaction_rows = [x for x in csv_reader]
row_str = ' '.join(transaction_rows[user_response][:-1])


#server
proba_threshold = 0.4
file = open(path.join('models', 'rf.pkl'), 'rb')
clf_obj = pickle.load(file)
rf_model = clf_obj[0]
mask = clf_obj[1]

def infer(clf, mask,  row):
    features_all = [float(x) for x in row.split()]
    features = [x for idx, x in enumerate(features_all) if mask[idx]]
    X_test = np.array([np.array(features)])
    probs = clf.predict_proba(X_test)
    preds = probs[:, 1]

    y_pred = [1 if x >= proba_threshold else 0 for x in preds]

    response =str(y_pred[0])
    return response

infer(rf_model, mask, row_str)