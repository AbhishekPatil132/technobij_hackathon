from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('lung_cancer_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        inputs = [int(request.form[key]) for key in request.form.keys()]
        inputs = np.array(inputs).reshape(1, -1)
        prediction = model.predict(inputs)[0]
        result = "Positive " if prediction == 1 else "Negative "
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
