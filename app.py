from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__, static_url_path='/static')
CORS(app)

with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
    labEnc_sex = encoders['sex']
    labEnc_smoker = encoders['smoker']
    labEnc_region = encoders['region']

with open('column_transformer.pkl', 'rb') as f:
    ct = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        age = data.get('age')
        bmi = data.get('bmi')
        children = data.get('children')
        gender = data.get('gender')
        smoker = data.get('smoker')
        region = data.get('region')

        if age is None or bmi is None or children is None or gender is None or smoker is None or region is None:
            return jsonify({'error': 'All fields are required!'}), 400

        age = int(age)
        if age < 16 or age > 65:
            return jsonify({'error': 'Age must be between 16 and 65!'}), 400

        bmi = float(bmi)
        if bmi < 16 or bmi > 47:
            return jsonify({'error': 'BMI must be between 16 and 47!'}), 400

        children = int(children)
        if children < 0 or children > 5:
            return jsonify({'error': 'Number of children must be between 0 and 5!'}), 400

        if gender not in ['male', 'female']:
            return jsonify({'error': 'Invalid gender selected!'}), 400

        if smoker not in ['yes', 'no']:
            return jsonify({'error': 'Invalid smoker selection!'}), 400

        if region not in ['northeast', 'northwest', 'southeast', 'southwest']:
            return jsonify({'error': 'Invalid region selection!'}), 400

        x_custom = np.array([[age, gender, bmi, children, smoker, region]], dtype=object)

        x_custom[:, 1] = labEnc_sex.transform(x_custom[:, 1])
        x_custom[:, 4] = labEnc_smoker.transform(x_custom[:, 4])
        x_custom[:, 5] = labEnc_region.transform(x_custom[:, 5])

        x_custom = ct.transform(x_custom)

        prediction = model.predict(x_custom)
        return jsonify({'predicted_insurance_cost': round(prediction[0], 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
