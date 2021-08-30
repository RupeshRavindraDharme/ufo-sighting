import numpy as np
import pickle as pkl
from flask import Flask, request, render_template

app = Flask(__name__)

model = pkl.load(open('ufo-model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    value = prediction[0]
    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        'index.html',
        prediction_text = 'The predicted country of Appearance is {}'.format(countries[value])
    )

if __name__ == "__main__":
    app.run()
