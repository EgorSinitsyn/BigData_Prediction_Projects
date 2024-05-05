import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

def predict_single(X, model):
    y_pred = model.predict(X)
    pred = np.expm1(y_pred)
    return pred[0]

with open('cars-price_GB.bin', 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST']) # Назначает маршрут /predict функции predict
def predict():
    data = request.get_json() # Получает содержимое запроса в Json

    # Преобразует данные из JSON в DataFrame
    X = pd.DataFrame.from_dict(data)

    # Оценивает клиента
    prediction = predict_single(X, model)

    # Подготавливает ответ
    result = {
        'price_prediction_HGB': float(prediction),  # Преобразует результат в float
    }

    # Преобразует в json
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
