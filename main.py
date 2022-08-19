# Melroy pereira


# Importing libraries
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
from flask import Flask,render_template,request,jsonify




    # app
app = Flask(__name__)

# model
model = pickle.load(open('model.pkl', 'rb'))

#scaler Y
scaling = pickle.load(open("scaler.pkl", 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    array_features = np.array([np.array(features)]).reshape(1,-1)
    prediction = model.predict(array_features)
    print(prediction)
    output = prediction[0]
    if output==1:
        output="Malignant"
    else:
        output="Benign"
    return render_template('index.html', prediction_text='The predicted cancer type is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)