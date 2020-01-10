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

def infer(model, row):
    response ='1'
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
                data = conn.recv(1024)
                response = infer(None, data.decode('utf-8'))
                if not data:
                    break
                conn.sendall(response.encode('utf-8'))