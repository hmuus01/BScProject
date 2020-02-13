import os
from os import abort

from flask import Flask, request, redirect, render_template, send_file, url_for
import pandas as pd
import socket
import csv
#from client import send_row

app = Flask(__name__)



HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

@app.route('/option', methods=["GET", "POST"])
def option():

    if request.method == "POST":
        print("Hello")
        selectValue = request.form.get('cars')
        print("This is  selectvalue = "+selectValue)
        #######################################################
        user_response = int(selectValue)
        data_file = os.path.join('data', 'credit.csv')
        with open(data_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            transaction_rows = [x for x in csv_reader]
        row_str = ' '.join(transaction_rows[user_response][:-1])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(row_str.encode("utf-8"))
            data = s.recv(1024)

        ##############################################
        print(data)

    return redirect(url_for('drop', response=data))


@app.route('/drop', methods=['GET', 'POST'])
def drop():
    path = os.path.join('data', 'credit.csv')
    response = request.args['response']
    data_df = pd.read_csv(path)

    num_transactions =data_df.shape[0]

    transactions = ['transaction ' + str(i)for i in range(1,num_transactions)]
    indexes = [i for i in range(num_transactions)]
    # server uses model to predict the legitimacy of the data


    return render_template('dropdown.html', colours=transactions, indexes=indexes, response=response)


if __name__ == '__main__':
    app.run()
