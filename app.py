from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn import preprocessing


#es = pickle.load(open('es.pkl', 'rb'))
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
    

    a = np.array([[int(data1), data2, data3, data4, data5, data6, data7, data8]],dtype=object)

    arr = pd.DataFrame(a,index=None,columns=['Age','Gender','family_history','benefits','care_options','anonymity','leave','work_interfere'])

    print(arr)

    le=LabelEncoder()
    arr['Gender']=le.fit_transform(arr['Gender'])
    # Scaling Age
    scaler = MinMaxScaler()
    arr['Age'] = scaler.fit_transform(arr[['Age']])
        
    print(arr)


    #pred = model.predict(a)

    
    pred=model.predict([[0.59090909,1,0,0,0,0,0,3]])
    
    
    print("predict "+str(pred))

    return render_template('after.html',data=pred)



if __name__ == "__main__":
    app.run(debug=True)


