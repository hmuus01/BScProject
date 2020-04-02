# #!/usr/bin/env python3
# import os
# import socket
# from io import StringIO
#
#
# import pandas as pd
# import pickle
#
# #from flaskapp import dropdown
#
# HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
# PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
#
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind((HOST, PORT))
#     s.listen()
#     conn, addr = s.accept()
#     with conn:
#         print('Connected by', addr)
#         # while True:
#         # data = conn.recv(1024)
#         # response = 'server says: '+data.decode("utf-8")
#         # print(data)
#         # if not data:
#         # break
#         # conn.sendall(response.encode())
#         while True:
#             # server reads file
#             data = conn.recv((409600))
#             # server  creates a data frame from file
#             #data_df = pd.read_csv(data)
#
#             #path = os.path.join('data', 'credit.csv')
#
#             #data_df = pd.read_csv(path)
#
#             s = str(data, 'utf-8')
#
#             d = StringIO(s)
#
#             df = pd.read_csv(d)
#             # server loads model
#
#             model = pickle.load(open('models/rf.pkl', 'rb'))
#             # server uses model to predict the legitimacy of the data
#             result = model.predict(df)
#             response = 'server says: Hi'
#             if not data:
#              break
#             #conn.sendall(response)
#             print(response)
#         print("Done Receiving")
#         conn.close()
#         # Open one recv.txt file in write mode
#         # file = open("recv.txt", "wb")
#         # print("\n Copied file name will be recv.txt at server side\n")
#
#         # Now we do not know when client will concatct server so server should be listening contineously
#         # while True:
#         #     # Send a hello message to client
#         #     msg = "\n\n|---------------------------------|\n Hi Client[IP address: " + addr[
#         #         0] + "], \n ֲֳ**Welcome to Server** \n -Server\n|---------------------------------|\n \n\n"
#         #     conn.send(msg.encode())
#         #
#         #     # Receive any data from client side
#         #     RecvData = conn.recv(1024)
#         #     while RecvData:
#         #         file.write(RecvData)
#         #         RecvData = conn.recv(1024)
#         #         print('server received')
#         #
#         #     # Close the file opened at server side once copy is completed
#         #     file.close()
#         #     print("\n File has been copied successfully \n")

#!/usr/bin/env python3
#https://realpython.com/python-sockets/
import socket
import pickle
from os import path
import numpy as np

models = ['rf', 'svc', 'lr']
proba_threshold = 0.5

def get_model(model_str):
    file = open(path.join('models', model_str+'.pkl'), 'rb')
    clf_obj = pickle.load(file)
    rf_model = clf_obj[0]
    mask = clf_obj[1]

    return rf_model, mask

def infer(row):
    #response_tokens=[]
    #for x in row.split():
    #    response_tokens.append(x)
    # row = 'x1 x2 x3'
    # row.split['x1', 'x2', 'x3']
    response_tokens = [x for x in row.split()]
    model_str = response_tokens[0]
    #Load the model
    clf, mask = get_model(models[int(model_str)])

    features_all = [float(x) for x in response_tokens[1:]]
    features = [x for idx, x in enumerate(features_all) if mask[idx]]
    print("Below is Features")
    print(features)
    print("Below is Mask")
    print(mask)
    # acquire the features in a numpy array data structure
    X_test = np.array([np.array(features)])

    # we pass the features array to the model to predict the probability of the label
    probs = clf.predict_proba(X_test)
    #preds stores just the probability of the fraudulent label
    preds = probs[:, 1]
    print(probs)
    print(preds)
    #stores 1 if it greater than the threshold or 0 if not
    y_pred = [1 if x >= proba_threshold else 0 for x in preds]

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
                print('-----------')
                print(data.decode('utf-8'))
                print('-----------')
                # 6 The server uses the chosen model to infer using the features as input
                response = infer(data.decode('utf-8'))
                print(response)
                # 7 The server sends the result ( 0 or 1 ) back to the client
                conn.sendall(response.encode('utf-8'))