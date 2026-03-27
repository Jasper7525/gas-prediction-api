from flask import Flask, request, jsonify
import joblib, pandas as pd, numpy as np, os

app = Flask(__name__)

# Absolute path so Render always finds it regardless of working directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gas_prediction_model.pkl')

model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"WARNING: Could not load model: {e}")
    print("Server will use math fallback for all predictions")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        scale_reading     = float(data['scale_reading_kg'])
        tare_weight       = float(data['tare_weight_kg'])
        cylinder_capacity = float(data['cylinder_capacity_kg'])
        rolling_avg       = float(data['rolling_7day_avg_kg_per_day'])
        days_inactive     = int(data['days_since_last_cook'])
        hour              = int(data.get('hour_of_day', 12))
        dow               = int(data.get('day_of_week', 0))
        month             = int(data.get('month', 1))

        gas_remaining = round(scale_reading - tare_weight, 4)

        # Empty cylinder
        if gas_remaining <= 0:
            return jsonify({
                'days_remaining': 0,
                'gas_remaining_kg': 0.0,
                'source': 'rule',
                'message': 'Cylinder is empty'
            })

        # Inactive user — return null so Android knows to show "inactive"
        if days_inactive >= 7 or rolling_avg <= 0.01:
            return jsonify({
                'days_remaining': None,
                'gas_remaining_kg': gas_remaining,
                'source': 'inactive',
                'message': 'No prediction — inactive'
            })

        # Try AI model
        if model is not None:
            try:
                gas_fill_pct = round(gas_remaining / cylinder_capacity, 4)
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
                days = float(model.predict(input_data)[0])
                days = round(max(0.0, min(days, 90.0)), 1)
                return jsonify({
                    'days_remaining'  : days,
                    'gas_remaining_kg': gas_remaining,
                    'source'          : 'ai',
                    'message'         : f'Approximately {days} days of gas remaining'
                })
            except Exception as e:
                print(f"Model prediction failed: {e} — falling back to math")

        # Server-side math fallback (model missing or crashed)
        days = round(gas_remaining / rolling_avg, 1)
        days = max(0.0, min(days, 90.0))
        return jsonify({
            'days_remaining'  : days,
            'gas_remaining_kg': gas_remaining,
            'source'          : 'server_math',
            'message'         : f'Approximately {days} days of gas remaining'
        })

    except KeyError as e:
        return jsonify({'error': f'Missing required field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'      : 'ok',
        'model_loaded': model is not None,
        'model_path'  : MODEL_PATH
    })


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name'    : 'Gas Cylinder Prediction API',
        'version' : '2.0',
        'status'  : 'running',
        'endpoints': {
            'predict': 'POST /predict',
            'health' : 'GET  /health'
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

### `requirements.txt` — replace yours
```
flask==2.3.3
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.4
joblib==1.3.2
gunicorn==21.2.0
werkzeug==2.3.7
