import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from flask import Flask,request,jsonify,render_template

application=Flask(__name__)
app=application

# import ridge regressor model and standard scaler pickle
ridge_model=pickle.load(open('pickle_files/lin_reg.pkl', 'rb'))
standard_scaler=pickle.load(open('pickle_files/scaler.pkl', 'rb'))

# route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['Get', 'Post'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_sc=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_sc)

        return render_template('index.html', result=result[0])
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)