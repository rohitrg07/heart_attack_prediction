from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("Heart_attack.lb")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # if request.method =='POST':
    # Retrieve form data
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trtbps = int(request.form['trtbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalachh = int(request.form['thalach'])
    exng = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slp = int(request.form['slope'])
    caa = int(request.form['ca'])
    thall = int(request.form['thal'])
    input_features=[[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]]

    # Prepare the input data for prediction
    # input_features = pd.DataFrame([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]],
                                #   columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    # Make prediction
    prediction = model.predict(input_features)
    risk_level = "Heart attack risk is low" if prediction[0] == 0 else "Heart attack risk is high"

    # Render the result template with prediction
    return render_template('result.html', prediction=risk_level)

if __name__ == '__main__':
    app.run(debug=True)
