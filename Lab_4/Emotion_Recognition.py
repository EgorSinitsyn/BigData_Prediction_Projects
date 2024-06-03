import pandas as pd
import os
import numpy as np
import cv2
from keras.preprocessing import image
import keras
from keras.models import model_from_json, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ZeroPadding2D

# Зарегистрируйте классы перед загрузкой модели
keras.utils.get_custom_objects().update({'Sequential': Sequential})

face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = model_from_json(open('/Users/oudzhi/PycharmProjects/BigData_Prediction/Lab_4/model_lab4.json', 'r').read())
model.load_weights('/Users/oudzhi/PycharmProjects/BigData_Prediction/Lab_4/model_lab4.h5')

# Подключение к веб-камере
cap = cv2.VideoCapture(0)

while True:
    sec, image = cap.read()
    if not sec:
        print("Не удалось получить кадр с камеры.")
        break

    # Конвертация изображения в оттенки серого
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детекция лиц
    faces_detected = face_cascade_db.detectMultiScale(converted_image)

    for (x, y, w, h) in faces_detected:
        # Рисование прямоугольников вокруг обнаруженных лиц
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Обработка области лица
        face_gray = converted_image[y:y + h, x:x + w]
        face_gray = cv2.resize(face_gray, (48, 48))

        # Подготовка изображения для модели
        image_pixels = np.array(face_gray)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels = image_pixels.reshape(image_pixels.shape[0], 48, 48, 1)

        # Прогноз эмоций
        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions[0])

        # Распознавание эмоций
        emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion_prediction = emotion_detection[max_index]

        # Отображение текста с эмоцией
        cv2.putText(image, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Изменение размера изображения для отображения
    resize_image = cv2.resize(image, (1000, 600))
    cv2.imshow('My_emotion', resize_image)

    # Выход из цикла при нажатии клавиши 'b'
    if cv2.waitKey(10) == ord('b'):
        break

cap.release()
cv2.destroyAllWindows()
print('success')
