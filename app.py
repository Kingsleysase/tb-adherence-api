from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

# Load model artifacts
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

def classify_risk(probability):
    if probability >= 0.70:
        return 'HIGH RISK'
    elif probability >= 0.40:
        return 'MEDIUM RISK'
    else:
        return 'LOW RISK'

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'TB Adherence API is running',
        'version': '1.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        categorical_cols = [
            'sex', 'location', 'education_level', 'employment_status',
            'household_income_level', 'tb_type', 'drug_resistance_status',
            'treatment_regimen', 'genexpert_result', 'sputum_smear_grade'
        ]
        continuous_cols = [
            'age', 'distance_to_facility_km', 'appointment_attendance_rate',
            'vot_submission_rate', 'merm_pillbox_adherence_rate',
            'sms_response_rate', 'avg_days_between_doses'
        ]

        # Encode categoricals
        for col in categorical_cols:
            if col in data and col in le_dict:
                try:
                    data[col] = int(le_dict[col].transform([data[col]])[0])
                except:
                    data[col] = 0

        # Build feature vector
        input_vector = [data.get(col, 0) for col in feature_columns]
        input_array = np.array(input_vector).reshape(1, -1)

        # Scale continuous features
        cont_indices = [feature_columns.index(c) for c in continuous_cols if c in feature_columns]
        input_array[:, cont_indices] = scaler.transform(
            input_array[:, cont_indices]
        )

        # Predict
        probability = float(model.predict_proba(input_array)[0][1])
        risk_tier = classify_risk(probability)

        return jsonify({
            'probability': round(probability, 3),
            'risk_tier': risk_tier,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/patients', methods=['GET'])
def get_sample_patients():
    patients = [
        {'id': 'PT0001', 'name': 'Patient 001', 'age': 34, 'risk_tier': 'HIGH RISK', 'probability': 0.82},
        {'id': 'PT0002', 'name': 'Patient 002', 'age': 45, 'risk_tier': 'MEDIUM RISK', 'probability': 0.55},
        {'id': 'PT0003', 'name': 'Patient 003', 'age': 28, 'risk_tier': 'LOW RISK', 'probability': 0.21},
        {'id': 'PT0004', 'name': 'Patient 004', 'age': 52, 'risk_tier': 'HIGH RISK', 'probability': 0.76},
        {'id': 'PT0005', 'name': 'Patient 005', 'age': 39, 'risk_tier': 'LOW RISK', 'probability': 0.18},
    ]
    return jsonify({'patients': patients, 'total': len(patients)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
