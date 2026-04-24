# shadowhunt
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Hackathon-КАМАЗ%202023-4CAF50)]()

> Нейросеть на базе U-Net для семантической сегментации и локализации людей на изображениях в реальном времени. Проект разработан в рамках хакатона НТЦ «КАМАЗ» и занял призовое место.

![til](./out3.gif)

## Запуск

Для обучения модели необходимо запустить файл **main.py**. Набор данных находится в папке **date**, он был сгенерирован вручную. Инструменты для работы с набором данных называются **resizer.py** и **mask_creator.py**. для тестирования сети вы можете запустить файл **test.py**, а для тестирования в режиме реального времени вам необходимо запустить **Show.py**. Обученная модель хранится в папке **model**.

Вы можете скачать обученную модель по ссылке: https://drive.google.com/drive/folders/1ByMH0K_IVUE4aCRbaOH50WY1-Sgnbarj?usp=drive_link оба файла должны быть помещены в папку model в корневом каталоге

## Особенности

- **Архитектура**: U-Net с skip connections для точной локализации границ
- **Функция потерь**: Комбинация Dice Loss + Binary Crossentropy для баланса точности и стабильности
- **Метрика**: Dice coefficient для оценки качества сегментации
- **Аугментация**: Случайные повороты (±20°), отражения для улучшения обобщающей способности
- **Оптимизация**: tf.data pipeline с кэшированием и предзагрузкой для эффективного обучения

## Пример датасета
| Входное изображение | Маска |
|--------------------|-------------------|
|<img src="data/peoples/video_2023-02-12_09-18-36_103.jpg">|<img src="data/masks/video_2023-02-12_09-18-36_103.jpg">|


## Пример локализации на датасете

| Входное изображение | Предсказание модели | Итог |
|---------------------|---------------------|------|
| <img width="280" height="284" alt="image" src="https://github.com/user-attachments/assets/1392ccf4-1e29-4557-9618-54f6149840d8" /> | <img width="275" height="273" alt="image" src="https://github.com/user-attachments/assets/bbb81c4e-a9a4-44c9-9163-e6d4ee6faff2" /> | <img width="276" height="279" alt="image" src="https://github.com/user-attachments/assets/249e1b00-d0a7-4a9a-ae54-6fffec728fa0" /> |
| <img width="278" height="275" alt="image" src="https://github.com/user-attachments/assets/ab568684-451a-42d1-882e-b0a6ba253312" /> | <img width="276" height="278" alt="image" src="https://github.com/user-attachments/assets/0c50e1b1-ec41-43a9-8005-7a9d35587932" /> | <img width="274" height="274" alt="image" src="https://github.com/user-attachments/assets/1d350be7-59ae-48de-a7e3-adf0fd87787a" /> |



## Архитектура:
![image](https://user-images.githubusercontent.com/80410524/218298737-c1eebc95-69ae-48e8-8963-c235b3c04730.png)
