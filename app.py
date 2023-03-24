from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn import preprocessing

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])

def home():
    data1 = request.form['age']
    data2 = request.form['gender']
    data3 = request.form['family']
    data4 = request.form['benefits']
    data5 = request.form['care']
    data6 = request.form['anonymous']
    data7 = request.form['leave']
    data8 = request.form['work']


    a = np.array([[data1, data2, data3, data4, data5, data6, data7, data8]])


    for i in range(8):
        le = preprocessing.LabelEncoder()
        le.fit(np.array([a[0][i]]))
        a[0][i] = int(le.transform(np.array([a[0][i]])))


    scaler = MinMaxScaler()
    a[0][0] = int(scaler.fit_transform(np.array([a[0][0]]).reshape(-1,1)))

    pred = model.predict(a)
    return render_template('after.html',data=pred)



if __name__ == "__main__":
    app.run(debug=True)


