import os
from os import abort
import random

from flask import Flask, request, redirect, render_template, send_file, url_for
import pandas as pd
import socket
import csv
#from client import send_row

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():

    return render_template('home.html')


HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

@app.route('/option', methods=["GET", "POST"])
def option():

    if request.method == "POST":
        print("Hello")
        selectValue = request.form.get('transaction')
        selectModel = request.form.get('model')
        print("This is  selectvalue = "+selectValue)
        print("This is  selectmodel = " + selectModel)
        #######################################################
        user_response = int(selectValue)
        data_file = os.path.join('data', 'credit.csv')
        with open(data_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            transaction_rows = [x for x in csv_reader]
        row_str = ' '.join(transaction_rows[user_response][:-1])
        send_str = ' '.join([selectModel, row_str])

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(send_str.encode("utf-8"))
            data = s.recv(1024).decode('utf-8')

        ##############################################
        print(data)

    return redirect(url_for('drop', response=str(data)+'_' + str(transaction_rows[user_response][-1])))


@app.route('/drop', methods=['GET', 'POST'])
def drop():
    path = os.path.join('data', 'credit.csv')
    response = request.args['response'].split('_')[0]
    true_val = request.args['response'].split('_')[1]
    data_df = pd.read_csv(path)

    num_transactions =data_df.shape[0]

    transactions = ['transaction ' + str(i)for i in range(1, num_transactions)]
    indexes = [i for i in range(random.randrange(num_transactions))]
    # server uses model to predict the legitimacy of the data

    models = ['rf', 'svm', 'lr']
    model_indexes = [i for i in range(len(models))]

    return render_template('dropdown.html', transactions=transactions, indexes=indexes, models=models,
                           model_indexes=model_indexes, response=response, true_val=true_val)

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
