import os
from os import abort

from flask import Flask, request, redirect, render_template, send_file
import pandas as pd

#from client import send_row

app = Flask(__name__)


#@app.route('/')
#def index():
    #return render_template('form.html')

# @app.route('/submit', methods=['POST'])
# def submit():
#     data = request.form['text']
#     response = send_to_server(data)
#     return 'You entered: {}'.format(response.decode("utf-8"))
#
# @app.route('/send', methods=['POST'])
# def send():
#     # obtain path of file
#     data = os.path.join('data', 'credit.csv')
#
#     # use path to send file over to server
#     response = sendFile(data)
#     return response

def hello():
    return "Hello World!"

@app.route('/option', methods=["GET", "POST"])
def option():

    if request.method == "POST":
        print("Hello")
        selectValue = request.form.get('cars')

        path = os.path.join('data', 'credit.csv')

        data_df = pd.read_csv(path)

        num_transactions = data_df.shape[0]

        transactions = ['transaction ' + str(i) for i in range(num_transactions)]
        indexes = [i for i in range(num_transactions)]


    return render_template("dropdown.html", colours=transactions, indexes=indexes)




@app.route('/upload', methods=["GET", "POST"])
def upload():

    if request.method == "POST":

        if request.files:

            path = os.path.join('data', 'credit.csv')

            return redirect(request.url)

    return render_template("upload_file.html")


@app.route('/drop', methods=['GET', 'POST'])
def drop():
    path = os.path.join('data', 'credit.csv')

    data_df = pd.read_csv(path)

    num_transactions =data_df.shape[0]

    transactions = ['transaction ' + str(i)for i in range(num_transactions)]
    indexes = [i for i in range(num_transactions)]
    # server uses model to predict the legitimacy of the data

    #send_row(rows)

    return render_template('dropdown.html', colours=transactions, indexes=indexes)



@app.route("/", methods=["GET", "POST"])
def upload_file():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            print("From upload statement" + image)

            return redirect(request.url)


    return render_template("upload_file.html")


if __name__ == '__main__':
    app.run()
