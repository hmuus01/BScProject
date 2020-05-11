import os
from os import abort
import random

from flask import Flask, request, redirect, render_template, send_file, url_for
import pandas as pd
import socket
import csv
#from client import send_row

app = Flask(__name__)

#1 The client: reads in a test dataset
path = os.path.join('data', 'validation_data.csv')
print('reading csv')
data_df = pd.read_csv(path, header=None)
print('shuffling')
data_df = data_df.sample(frac=1)
print(data_df)
#TEST Data

@app.route('/', methods=["GET", "POST"])
def home():

    return render_template('home.html')


HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server


@app.route('/option', methods=["GET", "POST"])
def option():
    # 3 The user chooses transaction and model
    if request.method == "POST":
        print("Hello")
        selectValue = request.form.get('transaction')
        selectModel = request.form.get('model')
        print("This is  selectvalue = " + selectValue)
        print("This is  selectmodel = " + selectModel)
        #######################################################
        #get the index of the transaction row
        user_response = int(selectValue)
        print("user_response = " + str(user_response))
        #path to datafile
        transaction_rows = data_df.values.tolist()
        print("Transaction_rows = " + str(transaction_rows))
         #concatenate features for row corresponding to user response)
        strings_to_send = transaction_rows[user_response][:-1]
        print("Strings to send = " + str(strings_to_send))
        row_str = ' '.join([str(x) for x in strings_to_send])
        print("row_str = " + str(row_str))
        # <m> <f1> <f2> <f3> <...> <fn>
        #concatenate the selected model to the corresponding
        send_str = ' '.join([selectModel, row_str])
        print("send_str = " + str(send_str))
        # 4 The client sends to the server, the transaction features and the model
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(send_str.encode("utf-8"))
            #8 receive result
            data = s.recv(1024).decode('utf-8')

        ##############################################
        print("data received = " + str(data))
        print("true val: " + str(transaction_rows[user_response][-1]))
    # 8 The client displays the result back to the user
    return redirect(url_for('drop', response=str(data)+'_' + str(int(transaction_rows[user_response][-1]))+'_' + str(transaction_rows[user_response][-2])))


@app.route('/drop', methods=['GET', 'POST'])
def drop():
    response = request.args['response'].split('_')[0]
    true_val = request.args['response'].split('_')[1]
    price = request.args['response'].split('_')[2]

    num_transactions = data_df.shape[0]
    print('length: ' + str(num_transactions))

    transactions = ['transaction ' + str(i)for i in range(1, num_transactions + 1)]
    indexes = [i for i in range(num_transactions)]
    # server uses model to predict the legitimacy of the data

    models = ['rf', 'svc2', 'lr2']
    model_indexes = [i for i in range(len(models))]

    # 2 The client displays the test dataset transactions to the user, along with a choice of three models
    return render_template('dropdown.html', transactions=transactions, indexes=indexes, models=models,
                           model_indexes=model_indexes, response=response, true_val=true_val, price=price)

# @app.route('/model_drop', methods=['GET', 'POST'])
# def model_dropdown():
#     response = request.args['response']
#     # server uses model to predict the legitimacy of the data
#
#     transactions = ['rf', 'svm', 'lr']
#     indexes = [i for i in range(len(transactions))]
#
#     # server uses model to predict the legitimacy of the data
#     return render_template('dropdown.html', colours=transactions, indexes=indexes, response=response)

if __name__ == '__main__':
    app.run()
