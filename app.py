import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open('diabetesML.pkl', 'rb'))

#print('result: {}'.format(model.predict([[6, 168, 72, 35, 0, 33.6, 0.6, 50]])))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    #print(final_features)
    prediction = model.predict(final_features)

    if prediction[0] == 1:
        result = 'Preson have diabetes'
    else:
        result = 'Preson not have diabetes'

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
    