from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load('model_saved')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validate the input data
    if not all(key in data for key in ('type', 'amount', 'old_balance', 'new_balance')):
        return jsonify({'error': 'Invalid input data'}), 400

    # Convert the input data to the appropriate data types
    type_value = int(data['type'])
    amount = float(data['amount'])
    old_balance = float(data['old_balance'])
    new_balance = float(data['new_balance'])

    # Prepare the input data for the model
    input_data = np.array([[type_value, amount, old_balance, new_balance]])

    # Make a prediction
    prediction = model.predict(input_data)
    
    return jsonify({'prediction': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)