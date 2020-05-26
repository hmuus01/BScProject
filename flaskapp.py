#This file serves as the client, it is made with flask and recieves data from the server which is displayed to the user
#The server and the client communicate via sockets obtained and adapted the code for this from https://realpython.com/python-sockets/

#Import Statements
import os
from flask import Flask, request, redirect, render_template, url_for
import pandas as pd
import socket

app = Flask(__name__)

#1 The client: reads in a validation(test) dataset
path = os.path.join('data', 'validation_data.csv')
print('reading csv')
data_df = pd.read_csv(path, header=None)
print('shuffling')
data_df = data_df.sample(frac=1)
#Print the shuffled validation(test) dataset
print(data_df)

#obtained and adapted from https://realpython.com/python-sockets/
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

#Default route to the homepage
@app.route('/', methods=["GET", "POST"])
def home():
    #render the homepage
    return render_template('home.html')


@app.route('/option', methods=["GET", "POST"])
def option():
    # 3 The user chooses transaction and model
    #If the request is a post method do the following
    if request.method == "POST":

        #From the form get the select with option name - transaction
        selectTransaction = request.form.get('transaction')

        #From the form get the select with option name - model
        selectModel = request.form.get('model')

        #Print these indexes
        #print("This is the transaction NO. selected = " + selectTransaction)
        #print("This is the model index selected = " + selectModel)

        #######################################################

        #Cast the transaction index into an integer
        user_response = int(selectTransaction)
        #print("user_response = " + str(user_response))

        #convert the dataframe into a multi-dimensional list
        transaction_rows = data_df.values.tolist()
        #print("Transaction_rows = " + str(transaction_rows))

        #Store the row corresponding to user's selection without the output class)
        strings_to_send = transaction_rows[user_response][:-1]
        #print("Strings to send = " + str(strings_to_send))

        #Row string
        row_str = ' '.join([str(x) for x in strings_to_send])
        #print("row_str = " + str(row_str))

        #concatenate the selected model to the corresponding row selected by the user
        send_str = ' '.join([selectModel, row_str])
        #print("send_str = " + str(send_str))

        # 4 The client sends to the server, the transaction features and the model
        # obtained and adapted from https://realpython.com/python-sockets/
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(send_str.encode("utf-8"))
            #8 receive result(prediction&probability) from server
            data = s.recv(1024).decode('utf-8').split('_')

        #PRINT STATEMENTS during testing
        #print("result received = " + str(data[0]))
        #print("proba received = " + str(data[1]))
        #print("true val: " + str(transaction_rows[user_response][-1]))


    # 8 The client displays the result back to the user | redirect to the drop route where the result will be displayed
    return redirect(url_for('drop', response=str(data[0]) + '_' + str(int(transaction_rows[user_response][-1])) + '_' + str(transaction_rows[user_response][-2]))
                    +'_' + str(data[1])+'_' + selectTransaction +'_' + selectModel)


@app.route('/drop', methods=['GET', 'POST'])
def drop():
    #use request.args to access the incoming data
    #Response stores the models prediction
    response = request.args['response'].split('_')[0]
    #True val stores the correct output class (from the dataset)
    true_val = request.args['response'].split('_')[1]
    #amount stores the transaction amount
    amount = request.args['response'].split('_')[2]
    #prob stores the probability transaction was fraud (predicted by the classifier)
    prob = request.args['response'].split('_')[3]
    #selected_value stores the index of the transaction
    selected_value = request.args['response'].split('_')[4]
    #selected model stores the index of the selected model
    selected_model = request.args['response'].split('_')[5]

    #store the number of transactions
    num_transactions = data_df.shape[0]
    print('length: ' + str(num_transactions))

    #Create a list of the transactions with indexes to display on the client
    transactions = ['transaction ' + str(i)for i in range(1, num_transactions + 1)]
    indexes = [i for i in range(num_transactions)]

    #Model list to display on the client
    models = ['RF', 'SVM', 'LR']
    model_indexes = [i for i in range(len(models))]

    # 2 The client displays the test dataset transactions to the user, along with a choice of three models, amount, probility and prediction
    return render_template('dropdown.html', transactions=transactions, indexes=indexes, models=models,
                           model_indexes=model_indexes, response=response, true_val=true_val, amount=amount, prob=prob
                           , selected_idx=int(selected_value), selected_model=int(selected_model))

# Python assignes main to a script when its being executed | if the condition below is satisfied then the app will be executed
if __name__ == '__main__':
    app.run()
