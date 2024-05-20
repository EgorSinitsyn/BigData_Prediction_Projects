import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Загрузка дф
df = pd.read_csv('old_df.csv')

# Define Flask app
app = Flask('lab_2')

# Детализация названий кластеров
clusters = {
    "Работа и Новости": 0,
    "Новости": 1,
    "Магазин и Работа": 2,
    "Искусство": 3,
    "Программа": 4,
    "Работа": 5,
}

# Функция предсказания
@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    # Get data from request
    data = request.get_json()
    surname = data['Фамилия']
    name = data['Имя']

    # Поиск соответствия
    person = df[(df['Фамилия'] == surname) & (df['Имя'] == name)]

    if person.empty:
        return "Person not found in the dataset"
    else:
        cluster_name = person['Группа'].values[0]
        cluster_name_str = ""

        if cluster_name == 0:
            cluster_name_str = 'Работа и Новости'
        elif cluster_name == 1:
            cluster_name_str = 'Новости'
        elif cluster_name == 2:
            cluster_name_str = 'Магазин и Работа'
        elif cluster_name == 3:
            cluster_name_str = 'Искусство'
        elif cluster_name == 4:
            cluster_name_str = 'Программа'
        elif cluster_name == 5:
            cluster_name_str = 'Работа'

        return f'Рекомендации для пользователя: "{cluster_name_str}"'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
