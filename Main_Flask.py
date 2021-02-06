from flask import Flask, render_template, request
import pickle
import numpy as np

#Hepatitis Classifier DATA
modelSVC = pickle.load(open('SVC_Classifier.pkl', 'rb'))

#Concrete Regressor DATA
modelRFT = pickle.load(open('RFT_Regressor.pkl', 'rb'))

app = Flask(__name__)

var1 = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','b1','b2','b3','b4','b5','b6','b7','b8','b9','b10',]
var2 = ['a1','a2','a3','a4','a5','a6','a7','a8']

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/Hepatitis',methods=['POST'])
def hepatitis():
    return render_template('Hepatitis.html')

@app.route('/Concrete',methods=['POST'])
def concrete():
    return render_template('Concrete.html')

@app.route('/predict_hepatitis', methods=['POST'])
def hepatitis_data():
    data = []
    for x in var1:
        data.append(request.form[str(x)])

    print(data)
    arr = np.array([data])
    predSVC = modelSVC.predict(arr)

    if predSVC == 1:
        n1 = (1," Hepatitis Patient")
    else:
        n1 = (2," Normal Patient")

    return render_template('after.html', dataSVC=n1)

@app.route('/predict_concrete', methods=['POST'])
def concrete_data():
    data = []
    for x in var2:
        data.append(request.form[str(x)])

    arr = np.array([data])
    predRFT = modelRFT.predict(arr)

    predRFT = (predRFT[0],"Compressive Strength")
    return render_template('after.html', dataRFT=predRFT)

if __name__ == "__main__":
    app.run(debug=True)