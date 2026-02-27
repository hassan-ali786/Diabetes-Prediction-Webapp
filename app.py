from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model/diabetes_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
accuracy = pickle.load(open("model/accuracy.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', accuracy=round(accuracy*100,2))

@app.route('/predict', methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final = scaler.transform([features])

    prediction = model.predict(final)[0]
    probability = model.predict_proba(final)[0][prediction]

    if prediction == 1:
        result = "Diabetic"
        risk = round(probability*100,2)
    else:
        result = "Non-Diabetic"
        risk = round((1-probability)*100,2)

    return render_template(
        'result.html',
        result=result,
        risk=risk,
        accuracy=round(accuracy*100,2)
    )

if __name__ == "__main__":
    app.run(debug=True)