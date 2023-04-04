from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn import preprocessing


#le = pickle.load(open('es.pkl', 'rb'))
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

    arr = pd.DataFrame(a,index=None,columns=['Age', 'Gender','Country', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere','remote_work','tech_company','wellness_program'])


    xrr=pd.read_csv('intermediate.csv',index_col=0)

    #print(xrr)

    xrr=xrr[['Age', 'Gender','Country', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere','remote_work','tech_company','wellness_program']]

    #print(arr)

    for i in xrr:
        le=preprocessing.LabelEncoder()
        le.fit(xrr[i])
        if i=='Age':
            xrr[i]=le.transform(xrr[i])
        #print(xrr[i])
        arr[i]=le.transform(arr[i])



    #print(arr)
    # Scaling Age
    scaler = MinMaxScaler()
    scaler.fit(xrr[['Age']])
    arr['Age'] = scaler.transform(arr[['Age']])

    #print(type(arr[['Age']]))

    #es(arr,'Age')
        
    #print(arr)


    #pred = model.predict(a)

    
    pred=model.predict(arr)
    pred_proba=model.predict_proba(arr)[:,1]
    #pred_proba=pred_proba.reshape(-1,1)

    t_arr=pred_proba[0]
    if(t_arr>=0 and t_arr<0.2):
            data=0
    if(t_arr>=0.2 and t_arr<0.5):
            data=1
    else:
            data=2
    
    #print("predict "+str(pred))

    return render_template('after.html',data=data)



if __name__ == "__main__":
    app.run(debug=True)


