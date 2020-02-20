# #!/usr/bin/env python3
# import os
# import socket
#
# import pandas as pd
#
# HOST = '127.0.0.1'  # The server's hostname or IP address
# PORT = 65432        # The port used by the server
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect((HOST, PORT))
#
# path = os.path.join('data', 'credit.csv')
# df = pd.read_csv(path)
#
# def send_row(data):
#    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#    #s.connect((HOST, PORT))
#    s.sendall(data)
#    response = s.recv(409600)
#    return response
#    while True:
#        s.send(send_row())
#        #sendData = file.read(409600)
#        print('client send')
#        #Close the connection from client side
#        s.close()
#        print('done')
#
# def get_row():
#     # for row in df.itertuples():
#     # row = df.loc[0:]
#     # print(row)
#     num_transactions = df.shape[0]
#
#     #transactions = ['transaction ' + str(i) for i in range(num_transactions)]
#     indexes = [i for i in range(num_transactions)]
#     return indexes
#
# while True:
#     s.send(get_row())
#     break
# print("Done Sending")
# s.close()
#
#
#
# # def sendFile(filepath):
# #     # We can send file sample.txt
# #
# #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
# #         s.connect((HOST, PORT))
# #         file = open(filepath, "rb")
# #         SendData = file.read(1024)
# #         server_bytes = s.recv(1024)
# #         # Now we can receive data from server
# #         print("\n\n################## Below message is received from server ################## \n\n ", server_bytes)
#
#
#
# #sendFile(os.path.join('data', 'credit.csv'))
#
# ##
#
#https://realpython.com/python-sockets/
#!/usr/bin/env python3

import socket
import csv

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

user_response = 5

with open('data/credit.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    transaction_rows = [x for x in csv_reader]
row_str = ' '.join(transaction_rows[user_response][:-1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(row_str.encode("utf-8"))
    data = s.recv(1024)

print('Received', repr(data))