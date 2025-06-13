from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

print("✅ Step 1: Starting Flask app")

app = Flask(__name__)

print("✅ Step 2: Loading model...")
try:
    model = pickle.load(open('boston_model.pkl', 'rb'))
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)

@app.route('/')
def home():
    print("✅ Home page loaded.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("✅ Predict route triggered.")
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text=f'Predicted House Price: ${output}k')

if __name__ == '__main__':
    print("✅ Flask app is starting...")
    app.run(debug=True)