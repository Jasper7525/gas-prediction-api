from flask import Flask, request, jsonify
import joblib, pandas as pd, numpy as np, os

app = Flask(__name__)
model = joblib.load('gas_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        scale_reading     = float(data['scale_reading_kg'])
        tare_weight       = float(data['tare_weight_kg'])
        cylinder_capacity = float(data['cylinder_capacity_kg'])
        rolling_avg       = float(data['rolling_7day_avg_kg_per_day'])
        days_inactive     = int(data['days_since_last_cook'])
        hour              = int(data.get('hour_of_day', 20))
        dow               = int(data.get('day_of_week', 0))
        month             = int(data.get('month', 1))

        if days_inactive >= 7:
            return jsonify({
                'prediction': None,
                'message': 'No prediction — user inactive for 7+ days'
            })

        gas_remaining = round(scale_reading - tare_weight, 4)
        gas_fill_pct  = round(gas_remaining / cylinder_capacity, 4)

        input_data = pd.DataFrame([{
            'gas_remaining_kg'            : gas_remaining,
            'rolling_7day_avg_kg_per_day' : rolling_avg,
            'cylinder_capacity_kg'        : cylinder_capacity,
            'gas_fill_pct'                : gas_fill_pct,
            'days_since_last_cook'        : days_inactive,
            'hour_of_day'                 : hour,
            'day_of_week'                 : dow,
            'month'                       : month,
        }])

        days = round(float(model.predict(input_data)[0]), 1)
        days = max(0, days)

        return jsonify({
            'gas_remaining_kg' : gas_remaining,
            'days_remaining'   : days,
            'message'          : f'You have approximately {days} days of gas remaining'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Gas Prediction API is running 24/7 ✅'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name'    : 'Gas Cylinder Prediction API',
        'version' : '1.0',
        'status'  : 'running',
        'endpoint': '/predict  (POST)',
        'health'  : '/health   (GET)'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
