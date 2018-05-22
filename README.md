# Перенос древнерусского стиля

https://www.youtube.com/watch?v=4sVk68Rsjqg

## Особенности решения

1) Корпус параллельных русских и древнерусских текстов предобработан так, чтобы обеспечить их максимальное выравнивание.

2) Размер базы составил 15000 парных фраз. Максимальная длина 205 символов.

3) Использована слегка модифицированная нейросетевая архитектура ByteNet (https://arxiv.org/abs/1610.10099).

<img src="https://camo.githubusercontent.com/5ad89ba8ded314ba5fa4728d05debb958dbd601c/687474703a2f2f692e696d6775722e636f6d2f4945365a71366f2e6a7067"  width="300">

Исходные данные: https://www.dropbox.com/s/no4l1cs4h69tae8/original.zip?dl=0
Обученная модель: https://www.dropbox.com/s/yvxbd2m8qh6eo7d/conv1d_54?dl=0

## Требования

CUDA 8.0

Tensorlow >= 1.7

Keras >= 2.1

Python 3

## Примеры сгенерированных фраз

<img src="https://github.com/Ivanx32/NeuralTranslation/blob/master/example.png"  width="500">




