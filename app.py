import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tf.keras.models as model

app = Flask(__name__)
model = tf.keras.models.load_model('SalaryPredictor/')
scFeatures = pickle.load(open('FeatureTransformer.ft','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age = float(input("Enter age: "))
    sal = float(input("Enter sal: "))
    feature = np.array([[age,sal]])
    stdFeatures = scFeatures.transform(feature)
    predLabel = model.predict_classes(stdFeatures)

    return render_template('index.html', prediction_text='Given customer is a {} customer'.format('Good' if predLabel[0][0] == 1 else 'Bad'))


if __name__ == "__main__":
    app.run(debug=True)