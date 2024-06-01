import pickle
import pandas as pd
from flask import Flask, request, jsonify

with open('model_TESS_neural.bin', 'rb') as f_in:
    model_TESS_neural = pickle.load(f_in)
with open('model_SAVEE_neural.bin', 'rb') as f_in:
    model_SAVEE_neural = pickle.load(f_in)
with open('model_TESS_RandomForest.bin', 'rb') as f_in:
    model_TESS_RandomForest = pickle.load(f_in)
with open('model_SAVEE_RandomForest.bin', 'rb') as f_in:
    model_SAVEE_RandomForest = pickle.load(f_in)
with open('model_SAVEE_svm.bin', 'rb') as f_in:
    model_SAVEE_svm = pickle.load(f_in)
with open('model_TESS_svm.bin', 'rb') as f_in:
    model_TESS_svm = pickle.load(f_in)



# Функция для предсказания с выбранной моделью
def predict_model(X, model):
    return model.predict(X)

app = Flask('audio_predict')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_values = data.get('X', None)
    feature_names = data.get('feature_names', None)
    if X_values is None or feature_names is None:
        return jsonify({'error': 'Data or feature names are missing in the request'})
    X = pd.DataFrame(X_values, columns=feature_names)

    # Выбор модели
    model_name = data.get('model', None)
    if model_name is None:
        return jsonify({'error': 'Model name is missing in the request'})

    if model_name == 'model_TESS_neural':
        model = model_TESS_neural
    elif model_name == 'model_SAVEE_neural':
        model = model_SAVEE_neural
    elif model_name == 'model_TESS_RandomForest':
        model = model_TESS_RandomForest
    elif model_name == 'model_SAVEE_RandomForest':
        model = model_SAVEE_RandomForest
    elif model_name == 'model_SAVEE_svm':
        model = model_SAVEE_svm
    elif model_name == 'model_TESS_svm':
        model = model_TESS_svm
    else:
        return jsonify({'error': 'Invalid model name'})

    # Предсказание с выбранной моделью
    y_pred = predict_model(X, model)

    return jsonify({'predictions': y_pred.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
