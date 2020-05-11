
#!/usr/bin/env python3
#https://realpython.com/python-sockets/
import socket
import pickle
from os import path
import numpy as np

models = ['rf', 'svc2', 'lr2']
proba_threshold = 0.5

def get_model(model_str):
    file = open(path.join('models', model_str+'.pkl'), 'rb')
    clf_obj = pickle.load(file)
    print("clf_obj = " + str(clf_obj))
    model = clf_obj[0]
    print("model = " + str(model))
    mask = clf_obj[1]

    return model, mask

def infer(row):
    #response_tokens=[]
    #for x in row.split():
    #    response_tokens.append(x)
    # row = 'x1 x2 x3'
    # row.split['x1', 'x2', 'x3']
    response_tokens = [x for x in row.split()]
    print("response_tokens = " + str(response_tokens))
    model_str = response_tokens[0]
    #Load the model
    clf, mask = get_model(models[int(model_str)])

    features_all = [float(x) for x in response_tokens[1:]]
    print("features_all = " + str(features_all))
    features = [x for idx, x in enumerate(features_all) if mask[idx]]
    print("Below is Features//")
    print(features)
    print('//')
    print("Below is Mask#")
    print(mask)
    print('#')
    # acquire the features in a numpy array data structure
    X_test = np.array([np.array(features)])

    # pass the features array to the model to predict the probability of the label
    probs = clf.predict_proba(X_test)
    #preds stores just the probability of the fraudulent label
    preds = probs[:, 1]
    print("prob for both "+str(probs))
    print("Prob its a 1 " +str(preds))
    #stores 1 if it greater than the threshold or 0 if not
    y_pred = [1 if x >= proba_threshold else 0 for x in preds]
    print("y_pred " + str(y_pred))
    response =str(y_pred[0])
    return response

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                # 5 The server receives the transaction features and the model
                data = conn.recv(1024)
                if not data:
                    break
                print("What the server receives")
                print('-----------')
                print(data.decode('utf-8'))
                print("data decode = " + str(data.decode('utf-8')))
                print('-----------')
                # 6 The server uses the chosen model to infer using the features as input
                response = infer(data.decode('utf-8'))
                print("Response from server")
                print(response)
                # 7 The server sends the result ( 0 or 1 ) back to the client
                conn.sendall(response.encode('utf-8'))