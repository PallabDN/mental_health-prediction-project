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


@app.route('/form', methods=['POST'])
def form():
      return render_template('form.html')

@app.route('/predict', methods=['POST'])

def home():
    data1 = request.form['age']
    data2 = request.form['gender']
    data3 = request.form['country']
    data4 = request.form['family']
    data5 = request.form['benefits']
    data6 = request.form['care']
    data7 = request.form['anonymous']
    data8 = request.form['leave']
    data9 = request.form['work']
    data10 = request.form['remote']
    data11 = request.form['tech']
    data12 = request.form['wellness']
    

    a = np.array([[int(data1), data2, data3, data4, data5, data6, data7, data8,data9, data10,data11,data12]],dtype=object)

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


