# shadowhunt


Нейронная сеть, построенная на основе сверточной нейронной сети U-Net для локализации людей в реальном времени
![til](./out3.gif)

Для обучения модели необходимо запустить файл **main.py**. Набор данных находится в папке **date**, он был сгенерирован вручную. Инструменты для работы с набором данных называются **resizer.py** и **mask_creator.py**. для тестирования сети вы можете запустить файл **test.py**, а для тестирования в режиме реального времени вам необходимо запустить **Show.py**. Обученная модель хранится в папке **model**.

Вы можете скачать обученную модель по ссылке: https://drive.google.com/drive/folders/1ByMH0K_IVUE4aCRbaOH50WY1-Sgnbarj?usp=drive_link оба файла должны быть помещены в папку model в корневом каталоге

Пример фото из датасета:

<kbd>
  <img src="data/peoples/video_2023-02-12_09-18-36_103.jpg">
  <img src="data/masks/video_2023-02-12_09-18-36_103.jpg">
</kbd>

Пример локализации на датасете:

<kbd>
  <img src="https://user-images.githubusercontent.com/80410524/218298594-95f29aaa-0bed-4d4c-8ec9-9ad9c8131496.png">
  <img src="https://user-images.githubusercontent.com/80410524/218298600-eeb2de2c-802c-4dca-b44d-be5af348dfb1.png">
  <img src="https://user-images.githubusercontent.com/80410524/218298615-2f0269e5-4abf-4d1c-a768-9ac81603f22c.png">
</kbd>


# Архитектура:
![image](https://user-images.githubusercontent.com/80410524/218298737-c1eebc95-69ae-48e8-8963-c235b3c04730.png)
