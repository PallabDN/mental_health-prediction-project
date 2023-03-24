from flask import Flask, render_template, request
import pickle
import numpy as np

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
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)



if __name__ == "__main__":
    app.run(debug=True)


